"""Static capabilities, protocols, and exact scientific DAGs for history budget phase diagram."""

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

from .contracts import HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramMethodFreeze, HistoryBudgetPhaseDiagramPhase, HistoryBudgetPhaseDiagramRecurrenceResult, HistoryBudgetPhaseDiagramSeedRosterCommitment, HistoryBudgetPhaseDiagramTerminalCloseout
from .descriptors import decode_config
from .runtime_contracts import HistoryBudgetPhaseDiagramAdjudicationBundle, HistoryBudgetPhaseDiagramArrayManifest, HistoryBudgetPhaseDiagramBootstrapSummary, HistoryBudgetPhaseDiagramCanaryReport, HistoryBudgetPhaseDiagramDenominatorBundle, HistoryBudgetPhaseDiagramDevelopmentGate, HistoryBudgetPhaseDiagramDevelopmentLedger, HistoryBudgetPhaseDiagramEvaluationDesignFreeze, HistoryBudgetPhaseDiagramGeneratorBundle, HistoryBudgetPhaseDiagramHistoryBundle, HistoryBudgetPhaseDiagramNominationFreeze, HistoryBudgetPhaseDiagramObserverBundle, HistoryBudgetPhaseDiagramPhaseCloseout, HistoryBudgetPhaseDiagramRequestedUnitLedger, HistoryBudgetPhaseDiagramSeedRoster, HistoryBudgetPhaseDiagramUntouchedBundle, FLOAT64_ARRAY_PAYLOAD_SCHEMA, INT64_ARRAY_PAYLOAD_SCHEMA


VERSION = "1.0.0"
ARRAY_PAYLOAD_SCHEMA = FLOAT64_ARRAY_PAYLOAD_SCHEMA
INT_ARRAY_PAYLOAD_SCHEMA = INT64_ARRAY_PAYLOAD_SCHEMA
CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NPY_MEDIA_TYPE = "application/x-npy"

SEED_ROSTER_KEY = "history-budget-phase-diagram.seed-roster"
DENOMINATOR_KEY = "history-budget-phase-diagram.denominator-descriptor"
GENERATOR_KEY = "history-budget-phase-diagram.independent-generator"
DEVELOPMENT_GENERATOR_KEY = "history-budget-phase-diagram.development-generator"
HISTORY_KEY = "history-budget-phase-diagram.history-observer"
TARGETER_KEY = "history-budget-phase-diagram.boundary-targeter"
PREPARATION_SAMPLER_KEY = "history-budget-phase-diagram.preparation-sampler"
NOMINATION_FREEZE_KEY = "history-budget-phase-diagram.challenge-freeze"
ROSTER_FREEZE_KEY = "history-budget-phase-diagram.roster-freeze"
UNIT_ADJUDICATOR_KEY = "history-budget-phase-diagram.unit-adjudicator"
DEVELOPMENT_ADJUDICATOR_KEY = "history-budget-phase-diagram.development-adjudicator"
RECURRENCE_KEY = "history-budget-phase-diagram.recurrence-synthesizer"
REPORTER_KEY = "history-budget-phase-diagram.reporter"
DEVELOPMENT_REPORTER_KEY = "history-budget-phase-diagram.development-reporter"
METHOD_FREEZE_KEY = "history-budget-phase-diagram.method-freeze"
DESIGN_QUALIFIER_KEY = "history-budget-phase-diagram.design-qualifier"
DEVELOPMENT_QUALIFIER_KEY = "history-budget-phase-diagram.development-qualifier"

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
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=8 * 1024**2,
    output_bytes=8 * 1024**2,
)
_OBSERVER = ResourceBudget(
    cpu_cores=2,
    memory_bytes=8 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=2700,
    source_scan_bytes=32 * 1024**2,
    output_bytes=32 * 1024**2,
)
_GENERATOR = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=5400,
    source_scan_bytes=32 * 1024**2,
    output_bytes=128 * 1024**2,
)
_SYNTHESIS = ResourceBudget(
    cpu_cores=2,
    memory_bytes=8 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=1800,
    source_scan_bytes=32 * 1024**2,
    output_bytes=64 * 1024**2,
)


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramResourceProfile:
    """Frozen per-task ceilings measured and accepted by the excluded C9 block."""

    profile_id: str
    small: ResourceBudget
    observer: ResourceBudget
    generator: ResourceBudget
    synthesis: ResourceBudget


HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE = HistoryBudgetPhaseDiagramResourceProfile(
    profile_id="history-budget-phase-diagram-excluded-resource-task-budget",
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
        (HistoryBudgetPhaseDiagramConfig.SCHEMA,),
        (HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA,),
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
                    HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, HistoryBudgetPhaseDiagramArrayManifest.SCHEMA, HistoryBudgetPhaseDiagramObserverBundle.SCHEMA)
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA,
                )
            )
        ),
        (HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,),
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
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramObserverBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(sorted((HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA, HistoryBudgetPhaseDiagramCanaryReport.SCHEMA))),
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramObserverBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, HistoryBudgetPhaseDiagramArrayManifest.SCHEMA, HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA)
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramObserverBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        (HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,),
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(sorted((HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA, HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA))),
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
                    HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA,
                )
            )
        ),
        tuple(sorted((HistoryBudgetPhaseDiagramBootstrapSummary.SCHEMA, HistoryBudgetPhaseDiagramRecurrenceResult.SCHEMA))),
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
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramBootstrapSummary.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramRecurrenceResult.SCHEMA,
                    HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramTerminalCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    HistoryBudgetPhaseDiagramCanaryReport.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA,
                    HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
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
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA,
                    HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRoster.SCHEMA,
                    HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA,
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
                    HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                    HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA,
                    HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA,
                    HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA,
                )
            )
        ),
        (HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
        _SMALL,
    ),
)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("ascii")).hexdigest()


def history_budget_phase_diagram_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities = tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=value.key,
                    capability_version=VERSION,
                    kind=value.kind,
                    config_schema=HistoryBudgetPhaseDiagramConfig.SCHEMA,
                    config_schema_sha256=_schema_sha256(HistoryBudgetPhaseDiagramConfig.SCHEMA),
                    input_schema_ids=value.inputs,
                    output_schema_ids=value.outputs,
                    permissions=value.permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=value.maximum_access,
                    resource_ceiling=value.budget,
                    deterministic=value.deterministic,
                    seed_required=False,
                    language_id="python",
                    runtime_id="cpython-numpy-scipy-history-budget-phase-diagram",
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
        registry_id="history-budget-phase-diagram-simulator-morphism-registry",
        capabilities=capabilities,
    )


def history_budget_phase_diagram_phase_registry(
    *,
    implementation_sha256: str,
    config: HistoryBudgetPhaseDiagramConfig,
) -> CapabilityRegistry:
    """Return the least-privilege registry frozen into one phase candidate."""

    complete = history_budget_phase_diagram_registry(implementation_sha256=implementation_sha256)
    definitions, _links = _phase_definitions(config)
    selected_keys = {value.key for value in definitions}
    capabilities = tuple(
        value for value in complete.capabilities if value.capability_key in selected_keys
    )
    if {value.capability_key for value in capabilities} != selected_keys:
        raise ValueError("history budget phase diagram phase registry lacks a selected capability")
    return CapabilityRegistry(
        registry_id=f"history-budget-phase-diagram-{config.phase.value.lower()}-registry",
        capabilities=capabilities,
    )


def config_ref(config: HistoryBudgetPhaseDiagramConfig) -> CapabilityConfigRef:
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
        _Output(f"{prefix}-array-manifest", HistoryBudgetPhaseDiagramArrayManifest.SCHEMA),
        _Output(f"{prefix}-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output(f"{prefix}-bundle", bundle_schema),
    )


def _phase_definitions(config: HistoryBudgetPhaseDiagramConfig) -> tuple[tuple[_Step, ...], tuple[_Link, ...]]:
    if config.phase is HistoryBudgetPhaseDiagramPhase.NOMINATION:
        nomination_steps = (
            _Step(
                "nomination-validate-design",
                ScientificStage.QUALIFY,
                DESIGN_QUALIFIER_KEY,
                (_Output("requested-unit-ledger", HistoryBudgetPhaseDiagramRequestedUnitLedger.SCHEMA),),
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
                    _Output("roster-commitment", HistoryBudgetPhaseDiagramSeedRosterCommitment.SCHEMA),
                    _Output("seed-roster", HistoryBudgetPhaseDiagramSeedRoster.SCHEMA),
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
                (_Output("roster-freeze-closeout", HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA),),
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
                    _Output("nomination-closeout", HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA),
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
    if config.phase is HistoryBudgetPhaseDiagramPhase.CANARY:
        canary_steps = (
            _Step(
                "canary-config-schema-conformance",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("contract-report", HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "canary-dense-observer",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("dense-observer-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-sparse-generator",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("sparse-generator-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "canary-structural-rank",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("structural-rank-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-discrete-rank",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("discrete-rank-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-conditioning",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("conditioning-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-untouched-sampler",
                ScientificStage.PREPARE,
                PREPARATION_SAMPLER_KEY,
                (_Output("untouched-sampler-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "canary-generator-observer-firewall",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("firewall-report", HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "canary-cross-implementation-conformance",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("numerical-conformance", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-excluded-resource",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("resource-canary", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "c10-canary-qualification",
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("canary-qualification", HistoryBudgetPhaseDiagramCanaryReport.SCHEMA),
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
            ("canary-structural-rank", "structural-rank-canary"),
            ("canary-discrete-rank", "discrete-rank-canary"),
            ("canary-conditioning", "conditioning-canary"),
            ("canary-untouched-sampler", "untouched-sampler-canary"),
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
            canary_links_list.append(
                _Link(
                    producer,
                    output_id,
                    "canary-cross-implementation-conformance",
                    output_id,
                    ScientificInputRole.QUALIFICATION,
                )
            )
        canary_links_list.extend(
            (
                _Link(
                    "canary-generator-observer-firewall",
                    "firewall-report",
                    "canary-cross-implementation-conformance",
                    "firewall-report",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "canary-config-schema-conformance",
                    "contract-report",
                    "canary-excluded-resource",
                    "contract-report",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "canary-cross-implementation-conformance",
                    "numerical-conformance",
                    "canary-excluded-resource",
                    "numerical-conformance",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "canary-cross-implementation-conformance",
                    "numerical-conformance",
                    "c10-canary-qualification",
                    "numerical-conformance",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "canary-excluded-resource",
                    "resource-canary",
                    "c10-canary-qualification",
                    "resource-canary",
                    ScientificInputRole.QUALIFICATION,
                ),
            )
        )
        canary_links = tuple(canary_links_list)
        return canary_steps, canary_links
    if config.phase is HistoryBudgetPhaseDiagramPhase.DEVELOPMENT:
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
                        (_Output("denominator-bundle", HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", HistoryBudgetPhaseDiagramObserverBundle.SCHEMA),
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
                                HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                            ),
                            _Output(
                                "untouched-float-arrays",
                                ARRAY_PAYLOAD_SCHEMA,
                                ArtifactProfile.NUMPY_NO_PICKLE,
                            ),
                            _Output(
                                "untouched-int-array-manifest",
                                HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                            ),
                            _Output(
                                "untouched-int-arrays",
                                INT_ARRAY_PAYLOAD_SCHEMA,
                                ArtifactProfile.NUMPY_NO_PICKLE,
                            ),
                            _Output("untouched-bundle", HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA),
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
                        (_Output("nomination-freeze", HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        DEVELOPMENT_GENERATOR_KEY,
                        _array_outputs("generator", HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.FALSIFY,
                        DEVELOPMENT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA),),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _SMALL,
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
                    (_Output("development-ledger", HistoryBudgetPhaseDiagramDevelopmentLedger.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    "development-correctness-power-resource-gate",
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (_Output("development-gate", HistoryBudgetPhaseDiagramDevelopmentGate.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    "f0-freeze-method",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("method-freeze", HistoryBudgetPhaseDiagramMethodFreeze.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    "f1-freeze-evaluation-design",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("evaluation-design-freeze", HistoryBudgetPhaseDiagramEvaluationDesignFreeze.SCHEMA),),
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
                        _Output("development-closeout", HistoryBudgetPhaseDiagramPhaseCloseout.SCHEMA),
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
    evaluation_steps: list[_Step] = []
    evaluation_links: list[_Link] = []
    for unit_id in config.unit_ids:
        tag = unit_id
        ids = {
            "descriptor": f"evaluation-descriptor.{tag}",
            "history": f"evaluation-history.{tag}",
            "target": f"evaluation-targeter.{tag}",
            "untouched": f"untouched-evaluation.{tag}",
            "freeze": f"evaluation-challenge-freeze.{tag}",
            "generator": f"evaluation-generator.{tag}",
            "adjudicate": f"evaluation-adjudicate.{tag}",
        }
        evaluation_steps.extend(
            (
                _Step(
                    ids["descriptor"],
                    ScientificStage.PREPARE,
                    DENOMINATOR_KEY,
                    (_Output("denominator-bundle", HistoryBudgetPhaseDiagramDenominatorBundle.SCHEMA),),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    ids["history"],
                    ScientificStage.DEVELOP,
                    HISTORY_KEY,
                    _array_outputs("history", HistoryBudgetPhaseDiagramHistoryBundle.SCHEMA),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _OBSERVER,
                ),
                _Step(
                    ids["target"],
                    ScientificStage.FALSIFY,
                    TARGETER_KEY,
                    _array_outputs("targeter", HistoryBudgetPhaseDiagramObserverBundle.SCHEMA),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
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
                            HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                        ),
                        _Output(
                            "untouched-float-arrays",
                            ARRAY_PAYLOAD_SCHEMA,
                            ArtifactProfile.NUMPY_NO_PICKLE,
                        ),
                        _Output(
                            "untouched-int-array-manifest",
                            HistoryBudgetPhaseDiagramArrayManifest.SCHEMA,
                        ),
                        _Output(
                            "untouched-int-arrays",
                            INT_ARRAY_PAYLOAD_SCHEMA,
                            ArtifactProfile.NUMPY_NO_PICKLE,
                        ),
                        _Output("untouched-bundle", HistoryBudgetPhaseDiagramUntouchedBundle.SCHEMA),
                    ),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _GENERATOR,
                ),
                _Step(
                    ids["freeze"],
                    ScientificStage.FREEZE,
                    NOMINATION_FREEZE_KEY,
                    (_Output("nomination-freeze", HistoryBudgetPhaseDiagramNominationFreeze.SCHEMA),),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    ids["generator"],
                    ScientificStage.ACQUIRE,
                    GENERATOR_KEY,
                    _array_outputs("generator", HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA),
                    OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _GENERATOR,
                ),
                _Step(
                    ids["adjudicate"],
                    ScientificStage.EVALUATE,
                    UNIT_ADJUDICATOR_KEY,
                    (_Output("adjudication-bundle", HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA),),
                    OutcomeAccess.EVALUATOR_REVEAL,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.REVEAL,
                    _SMALL,
                ),
            )
        )
        evaluation_links.extend(_unit_links(ids, evaluation=True))
    evaluation_steps.extend(
        (
            _Step(
                "recurrence-synthesize",
                ScientificStage.SYNTHESIZE,
                RECURRENCE_KEY,
                (
                    _Output("bootstrap-summary", HistoryBudgetPhaseDiagramBootstrapSummary.SCHEMA),
                    _Output("recurrence-result", HistoryBudgetPhaseDiagramRecurrenceResult.SCHEMA),
                ),
                OutcomeAccess.EVALUATION_REVEALED,
                VisibilityCeiling.OUTCOME_VISIBLE,
                BarrierKind.NONE,
                _SYNTHESIS,
            ),
            _Step(
                "terminal-report",
                ScientificStage.REPORT,
                REPORTER_KEY,
                (
                    _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                    _Output("terminal-closeout", HistoryBudgetPhaseDiagramTerminalCloseout.SCHEMA),
                ),
                OutcomeAccess.EVALUATION_REVEALED,
                VisibilityCeiling.OUTCOME_VISIBLE,
                BarrierKind.NONE,
                _SMALL,
            ),
        )
    )
    for unit_id in config.unit_ids:
        task_id = f"evaluation-adjudicate.{unit_id}"
        evaluation_links.append(
            _Link(
                task_id,
                "adjudication-bundle",
                "recurrence-synthesize",
                f"adjudication-{unit_id}",
                ScientificInputRole.OUTCOME,
            )
        )
    evaluation_links.extend(
        (
            _Link(
                "recurrence-synthesize",
                "bootstrap-summary",
                "terminal-report",
                "bootstrap-summary",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "recurrence-synthesize",
                "recurrence-result",
                "terminal-report",
                "recurrence-result",
                ScientificInputRole.OUTCOME,
            ),
        )
    )
    return tuple(evaluation_steps), tuple(evaluation_links)


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


def protocol_template(*, registry: CapabilityRegistry, config: HistoryBudgetPhaseDiagramConfig) -> ProtocolTemplate:
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
                    "history-budget-phase-diagram-seed-roster-exclusive"
                    if definition.step_id.startswith("n")
                    else "history-budget-phase-diagram-terminal-exclusive"
                    if definition.step_id.startswith("x")
                    else f"history-budget-phase-diagram-{definition.step_id}",
                ),
                barrier=definition.barrier,
                maximum_attempts=2,
                obligation_ids=(f"history-budget-phase-diagram-{definition.step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"history-budget-phase-diagram-{config.phase.value.lower()}-protocol",
        template_version=VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramExternalRecord:
    input_id: str
    record: CanonicalRecord
    role: ScientificInputRole
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling


def _validate_external_record_roster(
    config: HistoryBudgetPhaseDiagramConfig,
    records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...],
) -> None:
    expected_ids = {
        HistoryBudgetPhaseDiagramPhase.NOMINATION: (),
        HistoryBudgetPhaseDiagramPhase.CANARY: (),
        HistoryBudgetPhaseDiagramPhase.DEVELOPMENT: (
            "input.history-budget-phase-diagram.development.canary-qualification",
            "input.history-budget-phase-diagram.development.evaluation-config",
            "input.history-budget-phase-diagram.development.seed-roster-commitment",
        ),
        HistoryBudgetPhaseDiagramPhase.EVALUATION: (
            "input.history-budget-phase-diagram.evaluation.design-freeze",
            "input.history-budget-phase-diagram.evaluation.seed-roster",
        ),
    }[config.phase]
    if tuple(value.input_id for value in records) != expected_ids:
        raise ValueError("history budget phase diagram phase external record roster differs")
    if config.phase in {HistoryBudgetPhaseDiagramPhase.NOMINATION, HistoryBudgetPhaseDiagramPhase.CANARY}:
        return
    by_type = {type(value.record): value for value in records}
    if len(by_type) != len(records):
        raise ValueError("history budget phase diagram phase external record types repeat")
    if config.phase is HistoryBudgetPhaseDiagramPhase.DEVELOPMENT:
        if set(by_type) != {
            HistoryBudgetPhaseDiagramCanaryReport,
            HistoryBudgetPhaseDiagramConfig,
            HistoryBudgetPhaseDiagramSeedRosterCommitment,
        }:
            raise ValueError("history budget phase diagram development parent record types differ")
        canary = by_type[HistoryBudgetPhaseDiagramCanaryReport]
        evaluation_config = by_type[HistoryBudgetPhaseDiagramConfig]
        commitment_record = by_type[HistoryBudgetPhaseDiagramSeedRosterCommitment]
        assert isinstance(canary.record, HistoryBudgetPhaseDiagramCanaryReport)
        assert isinstance(evaluation_config.record, HistoryBudgetPhaseDiagramConfig)
        assert isinstance(commitment_record.record, HistoryBudgetPhaseDiagramSeedRosterCommitment)
        if (
            not canary.record.passed
            or evaluation_config.record.phase is not HistoryBudgetPhaseDiagramPhase.EVALUATION
            or evaluation_config.record.seed_roster_commitment_sha256
            != commitment_record.record.fingerprint()
            or (canary.role, canary.outcome_access, canary.visibility)
            != (
                ScientificInputRole.QUALIFICATION,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
            )
            or (
                evaluation_config.role,
                evaluation_config.outcome_access,
                evaluation_config.visibility,
            )
            != (
                ScientificInputRole.MODEL,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
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
            raise ValueError("history budget phase diagram development parent record contract differs")
        return
    if set(by_type) != {HistoryBudgetPhaseDiagramEvaluationDesignFreeze, HistoryBudgetPhaseDiagramSeedRoster}:
        raise ValueError("history budget phase diagram evaluation parent record types differ")
    design = by_type[HistoryBudgetPhaseDiagramEvaluationDesignFreeze]
    roster = by_type[HistoryBudgetPhaseDiagramSeedRoster]
    assert isinstance(design.record, HistoryBudgetPhaseDiagramEvaluationDesignFreeze)
    assert isinstance(roster.record, HistoryBudgetPhaseDiagramSeedRoster)
    seed_commitment = design.record.seed_roster_commitment
    roster_bytes = roster.record.canonical_bytes()
    if (
        design.record.evaluation_config_sha256 != config.fingerprint()
        or config.seed_roster_commitment_sha256 != seed_commitment.fingerprint()
        or tuple(value.unit_id for value in roster.record.entries) != config.unit_ids
        or sha256(roster_bytes).hexdigest() != seed_commitment.seed_payload_sha256
        or sha256(b"history-budget-phase-diagram-seed-roster-commitment\0" + roster_bytes).hexdigest()
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
        raise ValueError("history budget phase diagram evaluation parent record contract differs")


def scientific_graph(
    *,
    registry: CapabilityRegistry,
    config: HistoryBudgetPhaseDiagramConfig,
    protocol: ProtocolTemplate,
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...] = (),
) -> CandidateScientificGraph:
    _validate_external_record_roster(config, external_records)
    definitions, links = _phase_definitions(config)
    if {value.step_id for value in definitions} != {value.step_id for value in protocol.steps}:
        raise ValueError("history budget phase diagram protocol/phase definitions differ")
    steps = {value.step_id: value for value in protocol.steps}
    outputs = {
        (step.step_id, output.output_id): output
        for step in protocol.steps
        for output in step.outputs
    }
    config_external = HistoryBudgetPhaseDiagramExternalRecord(
        input_id=f"input.history-budget-phase-diagram.{config.phase.value.lower()}.config",
        record=config,
        role=ScientificInputRole.MODEL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    records = tuple(sorted((config_external, *external_records), key=lambda value: value.input_id))
    input_ids = tuple(value.input_id for value in records)
    if input_ids != tuple(sorted(set(input_ids))):
        raise ValueError("history budget phase diagram external records must be sorted and unique")
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
        graph_id=f"graph.history-budget-phase-diagram.{config.phase.value.lower()}",
        external_inputs=external_inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _external_edge(
    record: HistoryBudgetPhaseDiagramExternalRecord,
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
    phase: HistoryBudgetPhaseDiagramPhase,
    record: CanonicalRecord,
    protocol: ProtocolTemplate,
) -> tuple[tuple[str, str], ...]:
    task_ids = {value.step_id for value in protocol.steps}
    if phase is HistoryBudgetPhaseDiagramPhase.DEVELOPMENT:
        if isinstance(record, HistoryBudgetPhaseDiagramCanaryReport):
            return (("development-correctness-power-resource-gate", "canary-qualification"),)
        if isinstance(record, HistoryBudgetPhaseDiagramSeedRosterCommitment):
            return (("f1-freeze-evaluation-design", "seed-roster-commitment"),)
        if isinstance(record, HistoryBudgetPhaseDiagramConfig) and record.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
            return (("f1-freeze-evaluation-design", "evaluation-config"),)
    if phase is HistoryBudgetPhaseDiagramPhase.EVALUATION:
        if isinstance(record, HistoryBudgetPhaseDiagramSeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith("evaluation-descriptor.")
            )
        if isinstance(record, HistoryBudgetPhaseDiagramEvaluationDesignFreeze):
            consumers = tuple(
                task_id
                for task_id in sorted(task_ids)
                if task_id.startswith(
                    (
                        "evaluation-descriptor.",
                        "evaluation-history.",
                        "evaluation-targeter.",
                        "untouched-evaluation.",
                        "evaluation-challenge-freeze.",
                        "evaluation-generator.",
                        "evaluation-adjudicate.",
                    )
                )
                or task_id in {"recurrence-synthesize", "terminal-report"}
            )
            return tuple((value, "evaluation-design-freeze") for value in consumers)
    raise ValueError("history budget phase diagram external record has no exact phase consumer roster")


def study_template(
    *,
    registry: CapabilityRegistry,
    config: HistoryBudgetPhaseDiagramConfig,
    protocol: ProtocolTemplate,
    experiment: ExperimentSpec,
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...] = (),
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
        raise ValueError("history budget phase diagram phase must have exactly one terminal reporter")
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
        template_key=f"history-budget-phase-diagram.{config.phase.value.lower()}",
        template_version=VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.history-budget-phase-diagram.{config.phase.value.lower()}",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


def candidate_catalog(
    *, templates: tuple[StudyTemplate, ...], registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id="catalog.history-budget-phase-diagram-simulator-morphism-challenges",
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
class HistoryBudgetPhaseDiagramConfigDecoder:
    provider_key: str
    config: HistoryBudgetPhaseDiagramConfig
    provider_version: str = VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != HistoryBudgetPhaseDiagramConfig.SCHEMA:
            raise ValueError("history budget phase diagram config schema differs")
        if decode_config(payload) != self.config:
            raise ValueError("history budget phase diagram config differs from the exact phase registration")


def config_decoders(
    catalog: CandidateCapabilityCatalog,
    config: HistoryBudgetPhaseDiagramConfig,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return tuple(
        HistoryBudgetPhaseDiagramConfigDecoder(value.provider_key, config) for value in catalog.registrations
    )


__all__ = [
    "ARRAY_PAYLOAD_SCHEMA",
    "CONFIG_MEDIA_TYPE",
    "DEVELOPMENT_ADJUDICATOR_KEY",
    "DEVELOPMENT_GENERATOR_KEY",
    "DENOMINATOR_KEY",
    "GENERATOR_KEY",
    "HISTORY_KEY",
    'HistoryBudgetPhaseDiagramConfigDecoder',
    'HistoryBudgetPhaseDiagramExternalRecord',
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
    'history_budget_phase_diagram_registry',
    'history_budget_phase_diagram_phase_registry',
    'study_template',
    "protocol_template",
    "scientific_graph",
]
