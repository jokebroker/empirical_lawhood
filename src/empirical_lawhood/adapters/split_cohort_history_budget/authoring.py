"""Static capabilities, protocols, and exact scientific DAGs for split cohort history budget."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .contracts import SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetMethodFreeze, SplitCohortHistoryBudgetPhase, SplitCohortHistoryBudgetRecurrenceResult, SplitCohortHistoryBudgetSeedRosterCommitment, SplitCohortHistoryBudgetTContinuationGateRecord, SplitCohortHistoryBudgetTerminalCloseout
from .descriptors import decode_config
from .runtime_contracts import SplitCohortHistoryBudgetAdjudicationBundle, SplitCohortHistoryBudgetArrayManifest, SplitCohortHistoryBudgetBootstrapSummary, SplitCohortHistoryBudgetCanaryReport, SplitCohortHistoryBudgetDenominatorBundle, SplitCohortHistoryBudgetDevelopmentGate, SplitCohortHistoryBudgetDevelopmentLedger, SplitCohortHistoryBudgetEvaluationDesignFreeze, SplitCohortHistoryBudgetGeneratorBundle, SplitCohortHistoryBudgetHistoryBundle, SplitCohortHistoryBudgetNominationFreeze, SplitCohortHistoryBudgetObserverBundle, SplitCohortHistoryBudgetPhaseCloseout, SplitCohortHistoryBudgetRequestedUnitLedger, SplitCohortHistoryBudgetSeedRoster, SplitCohortHistoryBudgetUntouchedBundle, SplitCohortHistoryBudgetUntouchedFreeze, FLOAT64_ARRAY_PAYLOAD_SCHEMA, INT64_ARRAY_PAYLOAD_SCHEMA


VERSION = "1.0.0"
ARRAY_PAYLOAD_SCHEMA = FLOAT64_ARRAY_PAYLOAD_SCHEMA
INT_ARRAY_PAYLOAD_SCHEMA = INT64_ARRAY_PAYLOAD_SCHEMA
CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NPY_MEDIA_TYPE = "application/x-npy"

SEED_ROSTER_KEY = "split-cohort-history-budget.seed-roster"
DENOMINATOR_KEY = "split-cohort-history-budget.denominator-descriptor"
GENERATOR_KEY = "split-cohort-history-budget.independent-generator"
DEVELOPMENT_GENERATOR_KEY = "split-cohort-history-budget.development-generator"
HISTORY_KEY = "split-cohort-history-budget.history-observer"
TARGETER_KEY = "split-cohort-history-budget.boundary-targeter"
PREPARATION_SAMPLER_KEY = "split-cohort-history-budget.preparation-sampler"
NOMINATION_FREEZE_KEY = "split-cohort-history-budget.challenge-freeze"
ROSTER_FREEZE_KEY = "split-cohort-history-budget.roster-freeze"
UNIT_ADJUDICATOR_KEY = "split-cohort-history-budget.unit-adjudicator"
DEVELOPMENT_ADJUDICATOR_KEY = "split-cohort-history-budget.development-adjudicator"
RECURRENCE_KEY = "split-cohort-history-budget.recurrence-synthesizer"
REPORTER_KEY = "split-cohort-history-budget.reporter"
DEVELOPMENT_REPORTER_KEY = "split-cohort-history-budget.development-reporter"
METHOD_FREEZE_KEY = "split-cohort-history-budget.method-freeze"
DESIGN_QUALIFIER_KEY = "split-cohort-history-budget.design-qualifier"
DEVELOPMENT_QUALIFIER_KEY = "split-cohort-history-budget.development-qualifier"

_READ_WRITE = (
    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
)
_DEVELOPMENT = tuple(sorted((*_READ_WRITE, CapabilityPermission.READ_DEVELOPMENT)))
_EVALUATOR = tuple(
    sorted(
        (
            *_READ_WRITE,
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
    )
)
_REVEALED_REPORTER = tuple(
    sorted(
        (
            *_READ_WRITE,
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
        )
    )
)

_SMALL = ResourceBudget(
    cpu_cores=1,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=8 * 1024**2,
    output_bytes=8 * 1024**2,
)
_OBSERVER = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=1800,
    source_scan_bytes=32 * 1024**2,
    output_bytes=32 * 1024**2,
)
_GENERATOR = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=2700,
    source_scan_bytes=32 * 1024**2,
    output_bytes=48 * 1024**2,
)
_SYNTHESIS = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=1800,
    source_scan_bytes=32 * 1024**2,
    output_bytes=32 * 1024**2,
)


@dataclass(frozen=True, slots=True)
class SplitCohortHistoryBudgetResourceProfile:
    """Frozen per-task ceilings measured and accepted by the excluded C9 block."""

    profile_id: str
    small: ResourceBudget
    observer: ResourceBudget
    generator: ResourceBudget
    synthesis: ResourceBudget


SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE = SplitCohortHistoryBudgetResourceProfile(
    profile_id="split-cohort-history-budget-conformance-task-budget",
    small=_SMALL,
    observer=_OBSERVER,
    generator=_GENERATOR,
    synthesis=_SYNTHESIS,
)


@dataclass(frozen=True, slots=True)
class _CapabilityDefinition:
    key: str
    kind: CapabilityKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    permissions: tuple[CapabilityPermission, ...]
    maximum_access: OutcomeAccess
    budget: ResourceBudget
    deterministic: bool = True


_DEFINITIONS = (
    _CapabilityDefinition(
        DESIGN_QUALIFIER_KEY,
        CapabilityKind.NUMERICAL_QUALIFIER,
        (SplitCohortHistoryBudgetConfig.SCHEMA,),
        (SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA,),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
    ),
    _CapabilityDefinition(
        DEVELOPMENT_QUALIFIER_KEY,
        CapabilityKind.NUMERICAL_QUALIFIER,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        TARGETER_KEY,
        CapabilityKind.FALSIFIER,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, SplitCohortHistoryBudgetArrayManifest.SCHEMA, SplitCohortHistoryBudgetObserverBundle.SCHEMA)
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _OBSERVER,
    ),
    _CapabilityDefinition(
        PREPARATION_SAMPLER_KEY,
        CapabilityKind.SIMULATOR,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        DENOMINATOR_KEY,
        CapabilityKind.SIMULATOR,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetSeedRoster.SCHEMA,
                )
            )
        ),
        (SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
    ),
    _CapabilityDefinition(
        DEVELOPMENT_ADJUDICATOR_KEY,
        CapabilityKind.DISCREPANCY_ESTIMATOR,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetGeneratorBundle.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                    SplitCohortHistoryBudgetNominationFreeze.SCHEMA,
                    SplitCohortHistoryBudgetObserverBundle.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(sorted((SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA, SplitCohortHistoryBudgetCanaryReport.SCHEMA))),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _OBSERVER,
    ),
    _CapabilityDefinition(
        DEVELOPMENT_GENERATOR_KEY,
        CapabilityKind.SIMULATOR,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetNominationFreeze.SCHEMA,
                    SplitCohortHistoryBudgetObserverBundle.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetGeneratorBundle.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        GENERATOR_KEY,
        CapabilityKind.SIMULATOR,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetNominationFreeze.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, SplitCohortHistoryBudgetArrayManifest.SCHEMA, SplitCohortHistoryBudgetGeneratorBundle.SCHEMA)
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.EVALUATION_SEALED,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        HISTORY_KEY,
        CapabilityKind.LAW_IDENTIFIER,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _OBSERVER,
    ),
    _CapabilityDefinition(
        NOMINATION_FREEZE_KEY,
        CapabilityKind.TRANSFORM,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                    SplitCohortHistoryBudgetObserverBundle.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                    SplitCohortHistoryBudgetSeedRoster.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetNominationFreeze.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
    ),
    _CapabilityDefinition(
        ROSTER_FREEZE_KEY,
        CapabilityKind.TRANSFORM,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetSeedRoster.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        (SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
    ),
    _CapabilityDefinition(
        METHOD_FREEZE_KEY,
        CapabilityKind.TRANSFORM,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA,
                    SplitCohortHistoryBudgetMethodFreeze.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(sorted((SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA, SplitCohortHistoryBudgetMethodFreeze.SCHEMA))),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _SMALL,
    ),
    _CapabilityDefinition(
        RECURRENCE_KEY,
        CapabilityKind.HYPOTHESIS_SYNTHESIZER,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetMethodFreeze.SCHEMA,
                )
            )
        ),
        tuple(sorted((SplitCohortHistoryBudgetBootstrapSummary.SCHEMA, SplitCohortHistoryBudgetRecurrenceResult.SCHEMA))),
        _REVEALED_REPORTER,
        OutcomeAccess.EVALUATION_REVEALED,
        _SYNTHESIS,
    ),
    _CapabilityDefinition(
        REPORTER_KEY,
        CapabilityKind.REPORTER,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetBootstrapSummary.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetRecurrenceResult.SCHEMA,
                    SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetTerminalCloseout.SCHEMA,
                    ScientificAdjudicationRecord.SCHEMA,
                )
            )
        ),
        _REVEALED_REPORTER,
        OutcomeAccess.EVALUATION_REVEALED,
        _SYNTHESIS,
    ),
    _CapabilityDefinition(
        DEVELOPMENT_REPORTER_KEY,
        CapabilityKind.REPORTER,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA,
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetCanaryReport.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentGate.SCHEMA,
                    SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    ScientificAdjudicationRecord.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        SEED_ROSTER_KEY,
        CapabilityKind.SOURCE,
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA,
                    SplitCohortHistoryBudgetSeedRoster.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SplitCohortHistoryBudgetPhaseCloseout.SCHEMA,
                    SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA,
                    SplitCohortHistoryBudgetSeedRoster.SCHEMA,
                    SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
        deterministic=False,
    ),
    _CapabilityDefinition(
        UNIT_ADJUDICATOR_KEY,
        CapabilityKind.EVALUATOR,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                    SplitCohortHistoryBudgetConfig.SCHEMA,
                    SplitCohortHistoryBudgetDenominatorBundle.SCHEMA,
                    SplitCohortHistoryBudgetGeneratorBundle.SCHEMA,
                    SplitCohortHistoryBudgetHistoryBundle.SCHEMA,
                    SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA,
                    SplitCohortHistoryBudgetMethodFreeze.SCHEMA,
                    SplitCohortHistoryBudgetNominationFreeze.SCHEMA,
                    SplitCohortHistoryBudgetUntouchedBundle.SCHEMA,
                )
            )
        ),
        (SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA,),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
        _SMALL,
    ),
)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("ascii")).hexdigest()


def split_cohort_history_budget_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities = tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=value.key,
                    capability_version=VERSION,
                    kind=value.kind,
                    config_schema=SplitCohortHistoryBudgetConfig.SCHEMA,
                    config_schema_sha256=_schema_sha256(SplitCohortHistoryBudgetConfig.SCHEMA),
                    input_schema_ids=value.inputs,
                    output_schema_ids=value.outputs,
                    permissions=value.permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=value.maximum_access,
                    resource_ceiling=value.budget,
                    deterministic=value.deterministic,
                    seed_required=False,
                    language_id="python",
                    runtime_id="cpython-numpy-scipy-split-cohort-history-budget",
                    requires_clean_commit=True,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=(
                        "complete-independent-unit",
                        "generator-observer-firewall",
                        "no-actuation-no-network",
                        "receipt-first-external-artifacts",
                        "sealed-reveal-least-privilege",
                    ),
                    implementation_sha256=implementation_sha256,
                )
                for value in _DEFINITIONS
            ),
            key=lambda value: value.registry_id,
        )
    )
    return CapabilityRegistry(
        registry_id="split-cohort-history-budget-simulator-morphism-registry",
        capabilities=capabilities,
    )


def split_cohort_history_budget_phase_registry(
    *,
    implementation_sha256: str,
    config: SplitCohortHistoryBudgetConfig,
) -> CapabilityRegistry:
    """Return the least-privilege registry frozen into one phase candidate."""

    complete = split_cohort_history_budget_registry(implementation_sha256=implementation_sha256)
    definitions, _links = _phase_definitions(config)
    selected_keys = {value.key for value in definitions}
    capabilities = tuple(
        value for value in complete.capabilities if value.capability_key in selected_keys
    )
    if {value.capability_key for value in capabilities} != selected_keys:
        raise ValueError("split cohort history budget phase registry lacks a selected capability")
    return CapabilityRegistry(
        registry_id=f"split-cohort-history-budget-{config.phase.value.lower()}-registry",
        capabilities=capabilities,
    )


def config_ref(config: SplitCohortHistoryBudgetConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=config.config_id,
        config_schema=config.SCHEMA,
        config_schema_sha256=_schema_sha256(config.SCHEMA),
        content_sha256=config.fingerprint(),
        artifact_id=f"config-artifact.{config.config_id}",
    )


@dataclass(frozen=True, slots=True)
class _Output:
    output_id: str
    schema: str
    profile: ArtifactProfile = ArtifactProfile.CANONICAL_JSON


@dataclass(frozen=True, slots=True)
class _Step:
    step_id: str
    stage: ScientificStage
    key: str
    outputs: tuple[_Output, ...]
    access: OutcomeAccess
    visibility: VisibilityCeiling
    barrier: BarrierKind
    budget: ResourceBudget


@dataclass(frozen=True, slots=True)
class _Link:
    producer: str
    output_id: str
    consumer: str
    consumer_input: str
    role: ScientificInputRole


def _array_outputs(prefix: str, bundle_schema: str) -> tuple[_Output, ...]:
    return (
        _Output(f"{prefix}-array-manifest", SplitCohortHistoryBudgetArrayManifest.SCHEMA),
        _Output(f"{prefix}-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output(f"{prefix}-bundle", bundle_schema),
    )


def _phase_definitions(config: SplitCohortHistoryBudgetConfig) -> tuple[tuple[_Step, ...], tuple[_Link, ...]]:
    if config.phase is SplitCohortHistoryBudgetPhase.NOMINATION:
        nomination_steps = (
            _Step(
                "nomination-validate-design",
                ScientificStage.QUALIFY,
                DESIGN_QUALIFIER_KEY,
                (_Output("requested-unit-ledger", SplitCohortHistoryBudgetRequestedUnitLedger.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "nomination-generate-seed-roster",
                ScientificStage.PREPARE,
                SEED_ROSTER_KEY,
                (
                    _Output("roster-commitment", SplitCohortHistoryBudgetSeedRosterCommitment.SCHEMA),
                    _Output("seed-roster", SplitCohortHistoryBudgetSeedRoster.SCHEMA),
                ),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "nomination-seal-roster",
                ScientificStage.FREEZE,
                ROSTER_FREEZE_KEY,
                (_Output("roster-freeze-closeout", SplitCohortHistoryBudgetPhaseCloseout.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.FREEZE,
                _SMALL,
            ),
            _Step(
                "nomination-closeout",
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("nomination-closeout", SplitCohortHistoryBudgetPhaseCloseout.SCHEMA),
                    _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                ),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
        )
        nomination_links = (
            _Link(
                "nomination-validate-design",
                "requested-unit-ledger",
                "nomination-generate-seed-roster",
                "requested-unit-ledger",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "nomination-generate-seed-roster",
                "seed-roster",
                "nomination-seal-roster",
                "seed-roster",
                ScientificInputRole.DENOMINATOR,
            ),
            _Link(
                "nomination-generate-seed-roster",
                "roster-commitment",
                "nomination-seal-roster",
                "seed-roster-commitment",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "nomination-generate-seed-roster",
                "roster-commitment",
                "nomination-closeout",
                "seed-roster-commitment",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "nomination-seal-roster",
                "roster-freeze-closeout",
                "nomination-closeout",
                "roster-freeze-closeout",
                ScientificInputRole.PARENT_RECEIPT,
            ),
        )
        return nomination_steps, nomination_links
    if config.phase is SplitCohortHistoryBudgetPhase.CANARY:
        canary_steps = (
            _Step(
                "canary-config-schema-conformance",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("contract-report", SplitCohortHistoryBudgetPhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "canary-dense-observer",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("dense-observer-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-sparse-generator",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("sparse-generator-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-structural-rank",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("certificate-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-discrete-rank",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("direction-preparation-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-conditioning",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("rank-transition-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-untouched-sampler",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("efficiency-equivalence-canary", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-generator-observer-firewall",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("firewall-conformance", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-cross-implementation-conformance",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-00", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "canary-excluded-resource",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-01", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "c10-excluded-resource-canary",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-02", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "c11-excluded-resource-canary",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-03", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "c12-concurrent-resource-qualification",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("resource-qualification", SplitCohortHistoryBudgetCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "c13-canary-qualification",
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("canary-qualification", SplitCohortHistoryBudgetCanaryReport.SCHEMA),
                    _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                ),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
        )
        canary_links_list: list[_Link] = []
        for producer, output_id in (
            ("canary-dense-observer", "dense-observer-canary"),
            ("canary-sparse-generator", "sparse-generator-canary"),
            ("canary-structural-rank", "certificate-canary"),
            ("canary-discrete-rank", "direction-preparation-canary"),
            ("canary-conditioning", "rank-transition-canary"),
            ("canary-untouched-sampler", "efficiency-equivalence-canary"),
        ):
            canary_links_list.append(
                _Link(
                    producer,
                    output_id,
                    "canary-generator-observer-firewall",
                    output_id,
                    ScientificInputRole.QUALIFICATION,
                )
            )
        canary_links_list.extend(
            (
                *tuple(
                    _Link(
                        "canary-generator-observer-firewall",
                        "firewall-conformance",
                        f"c{index}-excluded-resource-canary"
                        if index > 8
                        else "canary-cross-implementation-conformance",
                        "firewall-report",
                        ScientificInputRole.QUALIFICATION,
                    )
                    for index in range(8, 12)
                ),
                *tuple(
                    _Link(
                        f"c{index}-excluded-resource-canary"
                        if index > 8
                        else "canary-cross-implementation-conformance",
                        f"resource-canary-{index - 8:02d}",
                        "c12-concurrent-resource-qualification",
                        f"resource-canary-{index - 8:02d}",
                        ScientificInputRole.QUALIFICATION,
                    )
                    for index in range(8, 12)
                ),
                _Link(
                    "canary-config-schema-conformance",
                    "contract-report",
                    "c13-canary-qualification",
                    "contract-report",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "canary-generator-observer-firewall",
                    "firewall-conformance",
                    "c13-canary-qualification",
                    "firewall-conformance",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "c12-concurrent-resource-qualification",
                    "resource-qualification",
                    "c13-canary-qualification",
                    "resource-qualification",
                    ScientificInputRole.QUALIFICATION,
                ),
            )
        )
        canary_links = tuple(canary_links_list)
        return canary_steps, canary_links
    if config.phase is SplitCohortHistoryBudgetPhase.DEVELOPMENT:
        development_steps: list[_Step] = []
        development_links: list[_Link] = []
        for unit_id in config.unit_ids:
            tag = unit_id
            ids = {
                "descriptor": f"development-descriptor.{tag}",
                "history": f"development-history.{tag}",
                "target": f"development-targeter.{tag}",
                "untouched": f"development-untouched.{tag}",
                "freeze": f"development-challenge-freeze.{tag}",
                "generator": f"development-generator.{tag}",
                "adjudicate": f"development-adjudicate.{tag}",
            }
            development_steps.extend(
                (
                    _Step(
                        ids["descriptor"],
                        ScientificStage.PREPARE,
                        DENOMINATOR_KEY,
                        (_Output("denominator-bundle", SplitCohortHistoryBudgetDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", SplitCohortHistoryBudgetHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", SplitCohortHistoryBudgetObserverBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["untouched"],
                        ScientificStage.PREPARE,
                        PREPARATION_SAMPLER_KEY,
                        (
                            _Output(
                                "untouched-float-array-manifest",
                                SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                            ),
                            _Output(
                                "untouched-float-arrays",
                                ARRAY_PAYLOAD_SCHEMA,
                                ArtifactProfile.NUMPY_NO_PICKLE,
                            ),
                            _Output(
                                "untouched-int-array-manifest",
                                SplitCohortHistoryBudgetArrayManifest.SCHEMA,
                            ),
                            _Output(
                                "untouched-int-arrays",
                                INT_ARRAY_PAYLOAD_SCHEMA,
                                ArtifactProfile.NUMPY_NO_PICKLE,
                            ),
                            _Output("untouched-bundle", SplitCohortHistoryBudgetUntouchedBundle.SCHEMA),
                        ),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["freeze"],
                        ScientificStage.FREEZE,
                        NOMINATION_FREEZE_KEY,
                        (_Output("nomination-freeze", SplitCohortHistoryBudgetNominationFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        DEVELOPMENT_GENERATOR_KEY,
                        _array_outputs("generator", SplitCohortHistoryBudgetGeneratorBundle.SCHEMA),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.FALSIFY,
                        DEVELOPMENT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA),),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                )
            )
            development_links.extend(_unit_links(ids, evaluation=False))
        adjudicators = tuple(f"development-adjudicate.{unit_id}" for unit_id in config.unit_ids)
        development_steps.extend(
            (
                _Step(
                    "development-complete-ledger",
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (_Output("development-ledger", SplitCohortHistoryBudgetDevelopmentLedger.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    "development-correctness-power-resource-gate",
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (_Output("development-gate", SplitCohortHistoryBudgetDevelopmentGate.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    "f0-freeze-method",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("method-freeze", SplitCohortHistoryBudgetMethodFreeze.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    "f1-freeze-evaluation-design",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("evaluation-design-freeze", SplitCohortHistoryBudgetEvaluationDesignFreeze.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    "f2-development-closeout",
                    ScientificStage.REPORT,
                    DEVELOPMENT_REPORTER_KEY,
                    (
                        _Output("development-closeout", SplitCohortHistoryBudgetPhaseCloseout.SCHEMA),
                        _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                    ),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
            )
        )
        for task_id in adjudicators:
            development_links.append(
                _Link(
                    task_id,
                    "adjudication-bundle",
                    "development-complete-ledger",
                    f"adjudication-{task_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        development_links.extend(
            (
                _Link(
                    "development-complete-ledger",
                    "development-ledger",
                    "development-correctness-power-resource-gate",
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "development-correctness-power-resource-gate",
                    "development-gate",
                    "f0-freeze-method",
                    "development-gate",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "development-complete-ledger",
                    "development-ledger",
                    "f0-freeze-method",
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "f0-freeze-method",
                    "method-freeze",
                    "f1-freeze-evaluation-design",
                    "method-freeze",
                    ScientificInputRole.MODEL,
                ),
                _Link(
                    "f1-freeze-evaluation-design",
                    "evaluation-design-freeze",
                    "f2-development-closeout",
                    "evaluation-design-freeze",
                    ScientificInputRole.QUALIFICATION,
                ),
            )
        )
        return tuple(development_steps), tuple(development_links)
    if config.phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
        steps: list[_Step] = []
        links: list[_Link] = []
        for unit_id in config.unit_ids:
            ids = {
                "descriptor": f"targeted-evaluation-descriptor.{unit_id}",
                "history": f"targeted-evaluation-history.{unit_id}",
                "target": f"targeted-evaluation-observer.{unit_id}",
                "freeze": f"targeted-evaluation-challenge-freeze.{unit_id}",
                "generator": f"targeted-evaluation-generator.{unit_id}",
                "adjudicate": f"targeted-evaluation-adjudicate.{unit_id}",
            }
            steps.extend(
                (
                    _Step(
                        ids["descriptor"],
                        ScientificStage.PREPARE,
                        DENOMINATOR_KEY,
                        (_Output("denominator-bundle", SplitCohortHistoryBudgetDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", SplitCohortHistoryBudgetHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", SplitCohortHistoryBudgetObserverBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["freeze"],
                        ScientificStage.FREEZE,
                        NOMINATION_FREEZE_KEY,
                        (_Output("nomination-freeze", SplitCohortHistoryBudgetNominationFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        GENERATOR_KEY,
                        _array_outputs("generator", SplitCohortHistoryBudgetGeneratorBundle.SCHEMA),
                        OutcomeAccess.EVALUATION_SEALED,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.EVALUATE,
                        UNIT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA),),
                        OutcomeAccess.EVALUATOR_REVEAL,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.REVEAL,
                        _OBSERVER,
                    ),
                )
            )
            links.extend(_target_unit_links(ids))
        steps.extend(
            (
                _Step(
                    "xt0-targeted-synthesize",
                    ScientificStage.SYNTHESIZE,
                    RECURRENCE_KEY,
                    (
                        _Output("bootstrap-summary", SplitCohortHistoryBudgetBootstrapSummary.SCHEMA),
                        _Output("recurrence-result", SplitCohortHistoryBudgetRecurrenceResult.SCHEMA),
                    ),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SYNTHESIS,
                ),
                _Step(
                    "xt1-targeted-closeout",
                    ScientificStage.REPORT,
                    REPORTER_KEY,
                    (
                        _Output("targeted-evaluation-continuation-gate", SplitCohortHistoryBudgetTContinuationGateRecord.SCHEMA),
                        _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                        _Output("targeted-closeout", SplitCohortHistoryBudgetTerminalCloseout.SCHEMA),
                    ),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SMALL,
                ),
            )
        )
        for unit_id in config.unit_ids:
            links.append(
                _Link(
                    f"targeted-evaluation-adjudicate.{unit_id}",
                    "adjudication-bundle",
                    "xt0-targeted-synthesize",
                    f"adjudication-{unit_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        links.extend(
            (
                _Link(
                    "xt0-targeted-synthesize",
                    "bootstrap-summary",
                    "xt1-targeted-closeout",
                    "bootstrap-summary",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "xt0-targeted-synthesize",
                    "recurrence-result",
                    "xt1-targeted-closeout",
                    "recurrence-result",
                    ScientificInputRole.OUTCOME,
                ),
            )
        )
        return tuple(steps), tuple(links)

    if config.phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
        steps = []
        links = []
        for unit_id in config.unit_ids:
            ids = {
                "untouched": f"untouched-evaluation-preparation.{unit_id}",
                "freeze": f"untouched-evaluation-freeze.{unit_id}",
                "generator": f"untouched-evaluation-generator.{unit_id}",
                "adjudicate": f"untouched-evaluation-adjudicate.{unit_id}",
            }
            steps.extend(
                (
                    _Step(
                        ids["untouched"],
                        ScientificStage.PREPARE,
                        PREPARATION_SAMPLER_KEY,
                        _untouched_outputs(),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["freeze"],
                        ScientificStage.FREEZE,
                        NOMINATION_FREEZE_KEY,
                        (_Output("untouched-freeze", SplitCohortHistoryBudgetUntouchedFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        GENERATOR_KEY,
                        _array_outputs("generator", SplitCohortHistoryBudgetGeneratorBundle.SCHEMA),
                        OutcomeAccess.EVALUATION_SEALED,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.EVALUATE,
                        UNIT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", SplitCohortHistoryBudgetAdjudicationBundle.SCHEMA),),
                        OutcomeAccess.EVALUATOR_REVEAL,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.REVEAL,
                        _OBSERVER,
                    ),
                )
            )
            links.extend(_untouched_unit_links(ids))
        steps.extend(
            (
                _Step(
                    "xu0-untouched-synthesize",
                    ScientificStage.SYNTHESIZE,
                    RECURRENCE_KEY,
                    (_Output("recurrence-result", SplitCohortHistoryBudgetRecurrenceResult.SCHEMA),),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SYNTHESIS,
                ),
                _Step(
                    "integrated-closeout",
                    ScientificStage.REPORT,
                    REPORTER_KEY,
                    (
                        _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                        _Output("terminal-closeout", SplitCohortHistoryBudgetTerminalCloseout.SCHEMA),
                    ),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SMALL,
                ),
            )
        )
        for unit_id in config.unit_ids:
            links.append(
                _Link(
                    f"untouched-evaluation-adjudicate.{unit_id}",
                    "adjudication-bundle",
                    "xu0-untouched-synthesize",
                    f"adjudication-{unit_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        links.append(
            _Link(
                "xu0-untouched-synthesize",
                "recurrence-result",
                "integrated-closeout",
                "recurrence-result",
                ScientificInputRole.OUTCOME,
            )
        )
        return tuple(steps), tuple(links)
    raise ValueError("unsupported split cohort history budget phase")


def _untouched_outputs() -> tuple[_Output, ...]:
    return (
        _Output("denominator-bundle", SplitCohortHistoryBudgetDenominatorBundle.SCHEMA),
        _Output("untouched-float-array-manifest", SplitCohortHistoryBudgetArrayManifest.SCHEMA),
        _Output("untouched-float-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output("untouched-int-array-manifest", SplitCohortHistoryBudgetArrayManifest.SCHEMA),
        _Output("untouched-int-arrays", INT_ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output("untouched-bundle", SplitCohortHistoryBudgetUntouchedBundle.SCHEMA),
    )


def _target_unit_links(ids: Mapping[str, str]) -> tuple[_Link, ...]:
    return (
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["history"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["target"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["target"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["freeze"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["target"],
            "targeter-bundle",
            ids["freeze"],
            "observer-bundle",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["generator"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["freeze"],
            "nomination-freeze",
            ids["generator"],
            "nomination-freeze",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["adjudicate"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["adjudicate"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["target"],
            "targeter-bundle",
            ids["adjudicate"],
            "observer-bundle",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["generator"],
            "generator-array-manifest",
            ids["adjudicate"],
            "generator-array-manifest",
            ScientificInputRole.OUTCOME,
        ),
        _Link(
            ids["generator"],
            "generator-arrays",
            ids["adjudicate"],
            "generator-arrays",
            ScientificInputRole.OUTCOME,
        ),
        _Link(
            ids["generator"],
            "generator-bundle",
            ids["adjudicate"],
            "generator-bundle",
            ScientificInputRole.OUTCOME,
        ),
    )


def _untouched_unit_links(ids: Mapping[str, str]) -> tuple[_Link, ...]:
    links = [
        _Link(
            ids["untouched"],
            "denominator-bundle",
            ids["generator"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["untouched"],
            "denominator-bundle",
            ids["adjudicate"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["untouched"],
            "untouched-bundle",
            ids["freeze"],
            "untouched-bundle",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["freeze"],
            "untouched-freeze",
            ids["generator"],
            "untouched-freeze",
            ScientificInputRole.MODEL,
        ),
    ]
    for output in (
        "untouched-bundle",
        "untouched-float-array-manifest",
        "untouched-float-arrays",
        "untouched-int-array-manifest",
        "untouched-int-arrays",
    ):
        links.append(
            _Link(
                ids["untouched"],
                output,
                ids["generator"],
                output,
                ScientificInputRole.PREPARED_MEDIUM,
            )
        )
    for output in ("untouched-bundle", "untouched-int-array-manifest", "untouched-int-arrays"):
        links.append(
            _Link(
                ids["untouched"],
                output,
                ids["adjudicate"],
                output,
                ScientificInputRole.PREPARED_MEDIUM,
            )
        )
    for output in ("generator-array-manifest", "generator-arrays", "generator-bundle"):
        links.append(
            _Link(ids["generator"], output, ids["adjudicate"], output, ScientificInputRole.OUTCOME)
        )
    return tuple(links)


def _unit_links(ids: Mapping[str, str], *, evaluation: bool) -> tuple[_Link, ...]:
    del evaluation
    return (
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["history"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["target"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["target"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["untouched"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["untouched"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["freeze"],
            "history-bundle",
            ScientificInputRole.HISTORY,
        ),
        _Link(
            ids["target"],
            "targeter-bundle",
            ids["freeze"],
            "observer-bundle",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["untouched"],
            "untouched-bundle",
            ids["freeze"],
            "untouched-bundle",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["generator"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["freeze"],
            "nomination-freeze",
            ids["generator"],
            "nomination-freeze",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["untouched"],
            "untouched-bundle",
            ids["generator"],
            "untouched-bundle",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-float-array-manifest",
            ids["generator"],
            "untouched-float-array-manifest",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-float-arrays",
            ids["generator"],
            "untouched-float-arrays",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-int-array-manifest",
            ids["generator"],
            "untouched-int-array-manifest",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-int-arrays",
            ids["generator"],
            "untouched-int-arrays",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["descriptor"],
            "denominator-bundle",
            ids["adjudicate"],
            "denominator-bundle",
            ScientificInputRole.DENOMINATOR,
        ),
        _Link(
            ids["history"],
            "history-bundle",
            ids["adjudicate"],
            "history-bundle",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["freeze"],
            "nomination-freeze",
            ids["adjudicate"],
            "nomination-freeze",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["target"],
            "targeter-bundle",
            ids["adjudicate"],
            "observer-bundle",
            ScientificInputRole.MODEL,
        ),
        _Link(
            ids["untouched"],
            "untouched-bundle",
            ids["adjudicate"],
            "untouched-bundle",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-int-array-manifest",
            ids["adjudicate"],
            "untouched-int-array-manifest",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["untouched"],
            "untouched-int-arrays",
            ids["adjudicate"],
            "untouched-int-arrays",
            ScientificInputRole.PREPARED_MEDIUM,
        ),
        _Link(
            ids["generator"],
            "generator-array-manifest",
            ids["adjudicate"],
            "generator-array-manifest",
            ScientificInputRole.OUTCOME,
        ),
        _Link(
            ids["generator"],
            "generator-arrays",
            ids["adjudicate"],
            "generator-arrays",
            ScientificInputRole.OUTCOME,
        ),
        _Link(
            ids["generator"],
            "generator-bundle",
            ids["adjudicate"],
            "generator-bundle",
            ScientificInputRole.OUTCOME,
        ),
    )


def protocol_template(*, registry: CapabilityRegistry, config: SplitCohortHistoryBudgetConfig) -> ProtocolTemplate:
    definitions, links = _phase_definitions(config)
    parents = {
        step.step_id: tuple(
            sorted({value.producer for value in links if value.consumer == step.step_id})
        )
        for step in definitions
    }
    ref = config_ref(config)
    steps = []
    for definition in definitions:
        manifest = registry.resolve(definition.key, VERSION)
        steps.append(
            ProtocolStepTemplate(
                step_id=definition.step_id,
                stage=definition.stage,
                capability_key=definition.key,
                capability_version=VERSION,
                config=ref,
                dependency_step_ids=parents[definition.step_id],
                outputs=tuple(
                    sorted(
                        (
                            OutputTemplate(
                                output_id=value.output_id,
                                payload_schema=value.schema,
                                profile=value.profile,
                                media_type=(
                                    NPY_MEDIA_TYPE
                                    if value.profile is ArtifactProfile.NUMPY_NO_PICKLE
                                    else CONFIG_MEDIA_TYPE
                                ),
                                filename_suffix=(
                                    ".npy"
                                    if value.profile is ArtifactProfile.NUMPY_NO_PICKLE
                                    else ".json"
                                ),
                            )
                            for value in definition.outputs
                        ),
                        key=lambda value: value.output_id,
                    )
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=definition.access,
                visibility_ceiling=definition.visibility,
                resource_budget=definition.budget,
                resource_lock_ids=(
                    "split-cohort-history-budget-seed-roster-exclusive"
                    if definition.step_id.startswith("n")
                    else "split-cohort-history-budget-terminal-exclusive"
                    if definition.step_id.startswith("x")
                    else f"split-cohort-history-budget-{definition.step_id}",
                ),
                barrier=definition.barrier,
                maximum_attempts=2,
                obligation_ids=(f"split-cohort-history-budget-{definition.step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"split-cohort-history-budget-{config.phase.value.lower()}-protocol",
        template_version=VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


@dataclass(frozen=True, slots=True)
class SplitCohortHistoryBudgetExternalRecord:
    input_id: str
    record: CanonicalRecord
    role: ScientificInputRole
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling


def _validate_external_record_roster(
    config: SplitCohortHistoryBudgetConfig,
    records: tuple[SplitCohortHistoryBudgetExternalRecord, ...],
) -> None:
    expected_ids: tuple[str, ...] = {
        SplitCohortHistoryBudgetPhase.NOMINATION: (),
        SplitCohortHistoryBudgetPhase.CANARY: (),
        SplitCohortHistoryBudgetPhase.DEVELOPMENT: (
            "input.split-cohort-history-budget.development.canary-qualification",
            "input.split-cohort-history-budget.development.targeted-evaluation-config",
            "input.split-cohort-history-budget.development.untouched-evaluation-config",
            "input.split-cohort-history-budget.development.seed-roster-commitment",
        ),
        SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION: (
            "input.split-cohort-history-budget.targeted-evaluation.design-freeze",
            "input.split-cohort-history-budget.targeted-evaluation.seed-roster",
        ),
        SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION: (
            "input.split-cohort-history-budget.untouched-evaluation.design-freeze",
            "input.split-cohort-history-budget.untouched-evaluation.seed-roster",
            "input.split-cohort-history-budget.untouched-evaluation.targeted-result",
            *tuple(
                f"input.split-cohort-history-budget.untouched-evaluation.descriptor.{unit_id}" for unit_id in config.unit_ids
            ),
        ),
    }[config.phase]
    if tuple(value.input_id for value in records) != expected_ids:
        raise ValueError("split cohort history budget phase external record roster differs")
    if config.phase in {SplitCohortHistoryBudgetPhase.NOMINATION, SplitCohortHistoryBudgetPhase.CANARY}:
        return
    if config.phase is SplitCohortHistoryBudgetPhase.DEVELOPMENT:
        canaries = tuple(
            value for value in records if isinstance(value.record, SplitCohortHistoryBudgetCanaryReport)
        )
        configs = tuple(value for value in records if isinstance(value.record, SplitCohortHistoryBudgetConfig))
        commitments = tuple(
            value for value in records if isinstance(value.record, SplitCohortHistoryBudgetSeedRosterCommitment)
        )
        if len(canaries) != 1 or len(configs) != 2 or len(commitments) != 1:
            raise ValueError("split cohort history budget development parent record types differ")
        canary = canaries[0]
        commitment_record = commitments[0]
        configs_by_phase = {value.record.phase: value for value in configs}
        if set(configs_by_phase) != {
            SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION,
            SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION,
        }:
            raise ValueError("split cohort history budget paired evaluation config phases differ")
        assert isinstance(canary.record, SplitCohortHistoryBudgetCanaryReport)
        assert isinstance(commitment_record.record, SplitCohortHistoryBudgetSeedRosterCommitment)
        if (
            not canary.record.passed
            or any(
                value.record.seed_roster_commitment_sha256 != commitment_record.record.fingerprint()
                for value in configs
            )
            or (canary.role, canary.outcome_access, canary.visibility)
            != (
                ScientificInputRole.QUALIFICATION,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
            )
            or any(
                (value.role, value.outcome_access, value.visibility)
                != (
                    ScientificInputRole.MODEL,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                )
                for value in configs
            )
            or (
                commitment_record.role,
                commitment_record.outcome_access,
                commitment_record.visibility,
            )
            != (
                ScientificInputRole.QUALIFICATION,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            )
        ):
            raise ValueError("split cohort history budget development parent record contract differs")
        return
    designs = tuple(
        value for value in records if isinstance(value.record, SplitCohortHistoryBudgetEvaluationDesignFreeze)
    )
    rosters = tuple(value for value in records if isinstance(value.record, SplitCohortHistoryBudgetSeedRoster))
    if len(designs) != 1 or len(rosters) != 1:
        raise ValueError("split cohort history budget evaluation design/roster record types differ")
    design = designs[0]
    roster = rosters[0]
    assert isinstance(design.record, SplitCohortHistoryBudgetEvaluationDesignFreeze)
    assert isinstance(roster.record, SplitCohortHistoryBudgetSeedRoster)
    seed_commitment = design.record.seed_roster_commitment
    roster_bytes = roster.record.canonical_bytes()
    if (
        (
            design.record.targeted_evaluation_config_sha256
            if config.phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION
            else design.record.untouched_evaluation_config_sha256
        )
        != config.fingerprint()
        or config.seed_roster_commitment_sha256 != seed_commitment.fingerprint()
        or tuple(value.unit_id for value in roster.record.entries) != config.unit_ids
        or sha256(roster_bytes).hexdigest() != seed_commitment.seed_payload_sha256
        or sha256(b"split-cohort-history-budget-seed-roster-commitment\0" + roster_bytes).hexdigest()
        != seed_commitment.commitment_sha256
        or (design.role, design.outcome_access, design.visibility)
        != (
            ScientificInputRole.MODEL,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
        )
        or (roster.role, roster.outcome_access, roster.visibility)
        != (
            ScientificInputRole.PREPARED_MEDIUM,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
    ):
        raise ValueError("split cohort history budget evaluation parent record contract differs")
    if config.phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
        targeted = tuple(
            value for value in records if isinstance(value.record, SplitCohortHistoryBudgetRecurrenceResult)
        )
        descriptors = tuple(
            value for value in records if isinstance(value.record, SplitCohortHistoryBudgetDenominatorBundle)
        )
        if (
            len(targeted) != 1
            or targeted[0].record.untouched_cells
            or len(descriptors) != 90
            or tuple(value.record.unit_id for value in descriptors) != config.unit_ids
            or any(
                tuple(value.scale_cells for value in descriptor.record.descriptors)
                != config.scale_cells
                for descriptor in descriptors
            )
            or any(
                value.record.preparation_substream_seed_sha256 == "0" * 64 for value in descriptors
            )
            or (targeted[0].role, targeted[0].outcome_access, targeted[0].visibility)
            != (
                ScientificInputRole.OUTCOME,
                OutcomeAccess.EVALUATION_REVEALED,
                VisibilityCeiling.OUTCOME_VISIBLE,
            )
            or any(
                (value.role, value.outcome_access, value.visibility)
                != (
                    ScientificInputRole.DENOMINATOR,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                )
                for value in descriptors
            )
        ):
            raise ValueError("split cohort history budget conditional U external records differ")


def scientific_graph(
    *,
    registry: CapabilityRegistry,
    config: SplitCohortHistoryBudgetConfig,
    protocol: ProtocolTemplate,
    external_records: tuple[SplitCohortHistoryBudgetExternalRecord, ...] = (),
) -> CandidateScientificGraph:
    _validate_external_record_roster(config, external_records)
    definitions, links = _phase_definitions(config)
    if {value.step_id for value in definitions} != {value.step_id for value in protocol.steps}:
        raise ValueError("split cohort history budget protocol/phase definitions differ")
    steps = {value.step_id: value for value in protocol.steps}
    outputs = {
        (step.step_id, output.output_id): output
        for step in protocol.steps
        for output in step.outputs
    }
    config_external = SplitCohortHistoryBudgetExternalRecord(
        input_id=f"input.split-cohort-history-budget.{config.phase.value.lower()}.config",
        record=config,
        role=ScientificInputRole.MODEL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    records = tuple(sorted((config_external, *external_records), key=lambda value: value.input_id))
    input_ids = tuple(value.input_id for value in records)
    if input_ids != tuple(sorted(set(input_ids))):
        raise ValueError("split cohort history budget external records must be sorted and unique")
    external_inputs = tuple(
        CandidateGraphExternalInput(
            input_id=value.input_id,
            scientific_role=value.role,
            logical_artifact_id=f"artifact.{value.input_id}",
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=value.record.fingerprint(),
            payload_schema=value.record.SCHEMA,
            media_type=CONFIG_MEDIA_TYPE,
            maximum_size_bytes=16 * 1024**2,
            outcome_access=value.outcome_access,
            visibility_ceiling=value.visibility,
        )
        for value in records
    )
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=registry.resolve(
                step.capability_key, VERSION
            ).implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    edges: list[CandidateGraphEdge] = []
    for link in links:
        output = outputs[(link.producer, link.output_id)]
        producer = steps[link.producer]
        consumer = steps[link.consumer]
        producer_output_access = (
            OutcomeAccess.EVALUATION_REVEALED
            if producer.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
            else producer.requested_outcome_access
        )
        producer_output_visibility = (
            VisibilityCeiling.OUTCOME_VISIBLE
            if producer.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
            else producer.visibility_ceiling
        )
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{link.producer}.{link.output_id}.{link.consumer}.{link.consumer_input}",
                producer_node_id=link.producer,
                producer_output_id=link.output_id,
                external_input_id=None,
                consumer_node_id=link.consumer,
                consumer_input_id=link.consumer_input,
                scientific_role=link.role,
                logical_artifact_id=f"artifact.{config.phase.value.lower()}.{link.producer}.{link.output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=producer.resource_budget.output_bytes,
                outcome_access=producer_output_access,
                visibility_ceiling=producer_output_visibility,
                barrier=consumer.barrier,
            )
        )
    for step in protocol.steps:
        edges.append(_external_edge(config_external, step, "phase-config"))
    for record in external_records:
        consumers = _external_consumers(config.phase, record.record, protocol)
        for consumer_id, consumer_input in consumers:
            edges.append(_external_edge(record, steps[consumer_id], consumer_input))
    return CandidateScientificGraph(
        graph_id=f"graph.split-cohort-history-budget.{config.phase.value.lower()}",
        external_inputs=external_inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _external_edge(
    record: SplitCohortHistoryBudgetExternalRecord,
    consumer: ProtocolStepTemplate,
    consumer_input: str,
) -> CandidateGraphEdge:
    return CandidateGraphEdge(
        edge_id=f"edge.external.{record.input_id}.{consumer.step_id}.{consumer_input}",
        producer_node_id=None,
        producer_output_id=None,
        external_input_id=record.input_id,
        consumer_node_id=consumer.step_id,
        consumer_input_id=consumer_input,
        scientific_role=record.role,
        logical_artifact_id=f"artifact.{record.input_id}",
        payload_schema=record.record.SCHEMA,
        media_type=CONFIG_MEDIA_TYPE,
        maximum_size_bytes=16 * 1024**2,
        outcome_access=record.outcome_access,
        visibility_ceiling=record.visibility,
        barrier=consumer.barrier,
    )


def _external_consumers(
    phase: SplitCohortHistoryBudgetPhase,
    record: CanonicalRecord,
    protocol: ProtocolTemplate,
) -> tuple[tuple[str, str], ...]:
    task_ids = {value.step_id for value in protocol.steps}
    if phase is SplitCohortHistoryBudgetPhase.DEVELOPMENT:
        if isinstance(record, SplitCohortHistoryBudgetCanaryReport):
            return (("development-correctness-power-resource-gate", "canary-qualification"),)
        if isinstance(record, SplitCohortHistoryBudgetSeedRosterCommitment):
            return (("f1-freeze-evaluation-design", "seed-roster-commitment"),)
        if isinstance(record, SplitCohortHistoryBudgetConfig) and record.phase in {
            SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION,
            SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION,
        }:
            return (
                (
                    "f1-freeze-evaluation-design",
                    "targeted-evaluation-config"
                    if record.phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION
                    else "untouched-evaluation-config",
                ),
            )
    if phase is SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION:
        if isinstance(record, SplitCohortHistoryBudgetSeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith("targeted-evaluation-descriptor.")
            )
        if isinstance(record, SplitCohortHistoryBudgetEvaluationDesignFreeze):
            consumers = tuple(
                task_id
                for task_id in sorted(task_ids)
                if task_id.startswith(
                    (
                        "evaluation-descriptor.",
                        "targeted-evaluation-descriptor.",
                        "targeted-evaluation-history.",
                        "targeted-evaluation-observer.",
                        "targeted-evaluation-challenge-freeze.",
                        "targeted-evaluation-generator.",
                        "targeted-evaluation-adjudicate.",
                    )
                )
                or task_id in {"xt0-targeted-synthesize", "xt1-targeted-closeout"}
            )
            return tuple((value, "evaluation-design-freeze") for value in consumers)
    if phase is SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION:
        if isinstance(record, SplitCohortHistoryBudgetEvaluationDesignFreeze):
            return tuple((task_id, "evaluation-design-freeze") for task_id in sorted(task_ids))
        if isinstance(record, SplitCohortHistoryBudgetSeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith("untouched-evaluation-preparation.")
            )
        if isinstance(record, SplitCohortHistoryBudgetDenominatorBundle):
            suffix = record.unit_id
            return ((f"untouched-evaluation-preparation.{suffix}", "denominator-bundle"),)
        if isinstance(record, SplitCohortHistoryBudgetRecurrenceResult):
            return (("xu0-untouched-synthesize", "targeted-result"),)
    raise ValueError("split cohort history budget external record has no exact phase consumer roster")


def study_template(
    *,
    registry: CapabilityRegistry,
    config: SplitCohortHistoryBudgetConfig,
    protocol: ProtocolTemplate,
    experiment: ExperimentSpec,
    external_records: tuple[SplitCohortHistoryBudgetExternalRecord, ...] = (),
) -> StudyTemplate:
    graph = scientific_graph(
        registry=registry,
        config=config,
        protocol=protocol,
        external_records=external_records,
    )
    protocol_owners = {
        obligation: (
            step.step_id,
            (
                "recurrence-result"
                if step.capability_key == RECURRENCE_KEY
                else step.outputs[0].output_id
            ),
        )
        for step in protocol.steps
        for obligation in step.obligation_ids
    }
    reporters = tuple(value for value in protocol.steps if value.stage is ScientificStage.REPORT)
    if len(reporters) != 1:
        raise ValueError("split cohort history budget phase must have exactly one terminal reporter")
    terminal = reporters[0]
    bindings = []
    for obligation in required_candidate_obligation_ids(experiment, protocol):
        owner_id, output_id = protocol_owners.get(
            obligation,
            (terminal.step_id, terminal.outputs[0].output_id),
        )
        contributors = tuple(
            sorted(value.edge_id for value in graph.edges if value.consumer_node_id == owner_id)
        )
        bindings.append(
            ObligationCoverageBinding(
                obligation_id=obligation,
                proof_owner_node_id=owner_id,
                required_output_id=output_id,
                contributor_edge_ids=contributors,
            )
        )
    return StudyTemplate(
        template_key=f"split-cohort-history-budget.{config.phase.value.lower()}",
        template_version=VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.split-cohort-history-budget.{config.phase.value.lower()}",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


def candidate_catalog(
    *, templates: tuple[StudyTemplate, ...], registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id="catalog.split-cohort-history-budget-simulator-morphism-challenges",
        registrations=tuple(
            CandidateCapabilityRegistration(
                manifest=value,
                provider_key=value.capability_key,
                provider_version=value.capability_version,
                config_media_type=CONFIG_MEDIA_TYPE,
                maximum_config_bytes=512 * 1024,
            )
            for value in registry.capabilities
        ),
        templates=tuple(sorted(templates, key=lambda value: value.template_key)),
    )


@dataclass(frozen=True, slots=True)
class SplitCohortHistoryBudgetConfigDecoder:
    provider_key: str
    config: SplitCohortHistoryBudgetConfig
    provider_version: str = VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SplitCohortHistoryBudgetConfig.SCHEMA:
            raise ValueError("split cohort history budget config schema differs")
        if decode_config(payload) != self.config:
            raise ValueError("split cohort history budget config differs from the exact phase registration")


def config_decoders(
    catalog: CandidateCapabilityCatalog,
    config: SplitCohortHistoryBudgetConfig,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return tuple(
        SplitCohortHistoryBudgetConfigDecoder(value.provider_key, config) for value in catalog.registrations
    )


__all__ = [
    "ARRAY_PAYLOAD_SCHEMA",
    "CONFIG_MEDIA_TYPE",
    "DEVELOPMENT_ADJUDICATOR_KEY",
    "DEVELOPMENT_GENERATOR_KEY",
    "DENOMINATOR_KEY",
    "GENERATOR_KEY",
    "HISTORY_KEY",
    'SplitCohortHistoryBudgetConfigDecoder',
    'SplitCohortHistoryBudgetExternalRecord',
    "NOMINATION_FREEZE_KEY",
    "RECURRENCE_KEY",
    "REPORTER_KEY",
    "SEED_ROSTER_KEY",
    "TARGETER_KEY",
    "UNIT_ADJUDICATOR_KEY",
    "VERSION",
    "candidate_catalog",
    "config_decoders",
    "config_ref",
    'split_cohort_history_budget_registry',
    'split_cohort_history_budget_phase_registry',
    'study_template',
    "protocol_template",
    "scientific_graph",
]
