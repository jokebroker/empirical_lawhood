"""Static capabilities, protocols, and exact scientific DAGs for receiver-history."""

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

from .contracts import (
    ReceiverHistoryConfig,
    ReceiverHistoryDevelopmentMetatheoryGate,
    ReceiverHistoryEmpiricalMetatheoryDossier,
    ReceiverHistoryInferenceDispositionMatrix,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryPhase,
    ReceiverHistoryRecurrenceResult,
    ReceiverHistorySeedRosterCommitment,
    ReceiverHistoryTargetedContinuationGateRecord,
    ReceiverHistoryTerminalCloseout,
)
from .descriptors import decode_config
from .runtime_contracts import (
    ReceiverHistoryAdjudicationBundle,
    ReceiverHistoryArrayManifest,
    ReceiverHistoryBootstrapSummary,
    ReceiverHistoryCanaryReport,
    ReceiverHistoryDenominatorBundle,
    ReceiverHistoryDevelopmentGate,
    ReceiverHistoryDevelopmentLedger,
    ReceiverHistoryEndpointCoordinatePowerAtlas,
    ReceiverHistoryEvaluationDesignFreeze,
    ReceiverHistoryGeneratorBundle,
    ReceiverHistoryHistoryBundle,
    ReceiverHistoryNominationFreeze,
    ReceiverHistoryObserverBundle,
    ReceiverHistoryPhaseCloseout,
    ReceiverHistoryRequestedUnitLedger,
    ReceiverHistorySeedRoster,
    ReceiverHistoryUntouchedBundle,
    ReceiverHistoryUntouchedFreeze,
    FLOAT64_ARRAY_PAYLOAD_SCHEMA,
    INT64_ARRAY_PAYLOAD_SCHEMA,
)


VERSION = "1.0.0"
ARRAY_PAYLOAD_SCHEMA = FLOAT64_ARRAY_PAYLOAD_SCHEMA
INT_ARRAY_PAYLOAD_SCHEMA = INT64_ARRAY_PAYLOAD_SCHEMA
CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NPY_MEDIA_TYPE = "application/x-npy"

SEED_ROSTER_KEY = "receiver-history.seed-roster"
DENOMINATOR_KEY = "receiver-history.denominator-source"
GENERATOR_KEY = "receiver-history.independent-generator"
DEVELOPMENT_GENERATOR_KEY = "receiver-history.development-generator"
HISTORY_KEY = "receiver-history.history-observer"
TARGETER_KEY = "receiver-history.endpoint-targeter"
PREPARATION_SAMPLER_KEY = "receiver-history.untouched-sampler"
NOMINATION_FREEZE_KEY = "receiver-history.challenge-freezer"
ROSTER_FREEZE_KEY = "receiver-history.roster-freeze"
UNIT_ADJUDICATOR_KEY = "receiver-history.unit-adjudicator"
DEVELOPMENT_ADJUDICATOR_KEY = "receiver-history.development-adjudicator"
RECURRENCE_KEY = "receiver-history.recurrence-synthesizer"
REPORTER_KEY = "receiver-history.reporter"
DEVELOPMENT_REPORTER_KEY = "receiver-history.development-reporter"
METHOD_FREEZE_KEY = "receiver-history.method-freeze"
DESIGN_QUALIFIER_KEY = "receiver-history.design-qualifier"
DEVELOPMENT_QUALIFIER_KEY = "receiver-history.development-qualifier"
ENDPOINT_POWER_KEY = "receiver-history.endpoint-power"

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
    wall_time_seconds=28800,
    source_scan_bytes=32 * 1024**2,
    output_bytes=32 * 1024**2,
)
_GENERATOR = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=28800,
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
class ReceiverHistoryResourceProfile:
    'Frozen per-task ceilings measured and accepted by the excluded target-correlated resource block.'

    profile_id: str
    small: ResourceBudget
    observer: ResourceBudget
    generator: ResourceBudget
    synthesis: ResourceBudget


RECEIVER_HISTORY_RESOURCE_PROFILE = ReceiverHistoryResourceProfile(
    profile_id='receiver-history-numerical-qualification-bounded-task-budget',
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
        (ReceiverHistoryConfig.SCHEMA,),
        (ReceiverHistoryRequestedUnitLedger.SCHEMA,),
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
                    ReceiverHistoryAdjudicationBundle.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDevelopmentLedger.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryDevelopmentGate.SCHEMA,
                    ReceiverHistoryDevelopmentLedger.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _GENERATOR,
    ),
    _CapabilityDefinition(
        ENDPOINT_POWER_KEY,
        CapabilityKind.ATLAS_ASSEMBLER,
        tuple(sorted((ReceiverHistoryConfig.SCHEMA, ReceiverHistoryDevelopmentLedger.SCHEMA))),
        (ReceiverHistoryEndpointCoordinatePowerAtlas.SCHEMA,),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        _SMALL,
    ),
    _CapabilityDefinition(
        TARGETER_KEY,
        CapabilityKind.FALSIFIER,
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryObserverBundle.SCHEMA,
                )
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    INT_ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistorySeedRoster.SCHEMA,
                )
            )
        ),
        (ReceiverHistoryDenominatorBundle.SCHEMA,),
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
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryGeneratorBundle.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                    ReceiverHistoryNominationFreeze.SCHEMA,
                    ReceiverHistoryObserverBundle.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted((ReceiverHistoryAdjudicationBundle.SCHEMA, ReceiverHistoryCanaryReport.SCHEMA))
        ),
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
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryNominationFreeze.SCHEMA,
                    ReceiverHistoryObserverBundle.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryGeneratorBundle.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryNominationFreeze.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryGeneratorBundle.SCHEMA,
                )
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                    ReceiverHistoryObserverBundle.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
                    ReceiverHistorySeedRoster.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ReceiverHistoryNominationFreeze.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistorySeedRoster.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        (ReceiverHistoryPhaseCloseout.SCHEMA,),
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDevelopmentGate.SCHEMA,
                    ReceiverHistoryDevelopmentLedger.SCHEMA,
                    ReceiverHistoryMethodFreeze.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ReceiverHistoryEvaluationDesignFreeze.SCHEMA, ReceiverHistoryMethodFreeze.SCHEMA)
            )
        ),
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
                    ReceiverHistoryAdjudicationBundle.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryMethodFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted((ReceiverHistoryBootstrapSummary.SCHEMA, ReceiverHistoryRecurrenceResult.SCHEMA))
        ),
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
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryBootstrapSummary.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistoryRecurrenceResult.SCHEMA,
                    ReceiverHistoryRequestedUnitLedger.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryDevelopmentGate.SCHEMA,
                    ReceiverHistoryDevelopmentMetatheoryGate.SCHEMA,
                    ReceiverHistoryDevelopmentLedger.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistoryTerminalCloseout.SCHEMA,
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
                    ReceiverHistoryAdjudicationBundle.SCHEMA,
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDevelopmentGate.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ReceiverHistoryCanaryReport.SCHEMA,
                    ReceiverHistoryDevelopmentGate.SCHEMA,
                    ReceiverHistoryDevelopmentMetatheoryGate.SCHEMA,
                    ReceiverHistoryDevelopmentLedger.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
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
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistoryRequestedUnitLedger.SCHEMA,
                    ReceiverHistorySeedRoster.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ReceiverHistoryPhaseCloseout.SCHEMA,
                    ReceiverHistoryRequestedUnitLedger.SCHEMA,
                    ReceiverHistorySeedRoster.SCHEMA,
                    ReceiverHistorySeedRosterCommitment.SCHEMA,
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
                    ReceiverHistoryArrayManifest.SCHEMA,
                    ReceiverHistoryConfig.SCHEMA,
                    ReceiverHistoryDenominatorBundle.SCHEMA,
                    ReceiverHistoryGeneratorBundle.SCHEMA,
                    ReceiverHistoryHistoryBundle.SCHEMA,
                    ReceiverHistoryEvaluationDesignFreeze.SCHEMA,
                    ReceiverHistoryMethodFreeze.SCHEMA,
                    ReceiverHistoryNominationFreeze.SCHEMA,
                    ReceiverHistoryUntouchedBundle.SCHEMA,
                )
            )
        ),
        (ReceiverHistoryAdjudicationBundle.SCHEMA,),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
        _SMALL,
    ),
)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("ascii")).hexdigest()


def receiver_history_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities = tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=value.key,
                    capability_version=VERSION,
                    kind=value.kind,
                    config_schema=ReceiverHistoryConfig.SCHEMA,
                    config_schema_sha256=_schema_sha256(ReceiverHistoryConfig.SCHEMA),
                    input_schema_ids=value.inputs,
                    output_schema_ids=value.outputs,
                    permissions=value.permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=value.maximum_access,
                    resource_ceiling=value.budget,
                    deterministic=value.deterministic,
                    seed_required=False,
                    language_id="python",
                    runtime_id="cpython-numpy-scipy-receiver-history",
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
        registry_id="receiver-history-simulator-morphism-registry",
        capabilities=capabilities,
    )


def receiver_history_phase_registry(
    *,
    implementation_sha256: str,
    config: ReceiverHistoryConfig,
) -> CapabilityRegistry:
    """Return the least-privilege registry frozen into one phase candidate."""

    complete = receiver_history_registry(implementation_sha256=implementation_sha256)
    definitions, _links = _phase_definitions(config)
    selected_keys = {value.key for value in definitions}
    capabilities = tuple(
        value for value in complete.capabilities if value.capability_key in selected_keys
    )
    if {value.capability_key for value in capabilities} != selected_keys:
        raise ValueError("receiver-history phase registry lacks a selected capability")
    return CapabilityRegistry(
        registry_id=f"receiver-history-{config.phase.value.lower()}-registry",
        capabilities=capabilities,
    )


def config_ref(config: ReceiverHistoryConfig) -> CapabilityConfigRef:
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
        _Output(f"{prefix}-array-manifest", ReceiverHistoryArrayManifest.SCHEMA),
        _Output(f"{prefix}-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output(f"{prefix}-bundle", bundle_schema),
    )


def _phase_definitions(
    config: ReceiverHistoryConfig,
) -> tuple[tuple[_Step, ...], tuple[_Link, ...]]:
    if config.phase is ReceiverHistoryPhase.NOMINATION:
        nomination_steps = (
            _Step(
                'nomination-validate-design',
                ScientificStage.QUALIFY,
                DESIGN_QUALIFIER_KEY,
                (_Output("requested-unit-ledger", ReceiverHistoryRequestedUnitLedger.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                'nomination-generate-seed-roster',
                ScientificStage.PREPARE,
                SEED_ROSTER_KEY,
                (
                    _Output("roster-commitment", ReceiverHistorySeedRosterCommitment.SCHEMA),
                    _Output("seed-roster", ReceiverHistorySeedRoster.SCHEMA),
                ),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                'nomination-seal-roster',
                ScientificStage.FREEZE,
                ROSTER_FREEZE_KEY,
                (_Output("roster-freeze-closeout", ReceiverHistoryPhaseCloseout.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.FREEZE,
                _SMALL,
            ),
            _Step(
                'nomination-closeout',
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("nomination-closeout", ReceiverHistoryPhaseCloseout.SCHEMA),
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
                'nomination-validate-design',
                "requested-unit-ledger",
                'nomination-generate-seed-roster',
                "requested-unit-ledger",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                'nomination-generate-seed-roster',
                "seed-roster",
                'nomination-seal-roster',
                "seed-roster",
                ScientificInputRole.DENOMINATOR,
            ),
            _Link(
                'nomination-generate-seed-roster',
                "roster-commitment",
                'nomination-seal-roster',
                "seed-roster-commitment",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                'nomination-generate-seed-roster',
                "roster-commitment",
                'nomination-closeout',
                "seed-roster-commitment",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                'nomination-seal-roster',
                "roster-freeze-closeout",
                'nomination-closeout',
                "roster-freeze-closeout",
                ScientificInputRole.PARENT_RECEIPT,
            ),
        )
        return nomination_steps, nomination_links
    if config.phase is ReceiverHistoryPhase.CANARY:
        canary_steps = (
            _Step(
                'qualification-profile-conformance',
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("profile-conformance", ReceiverHistoryPhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                'qualification-dense-equations',
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("dense-truth", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-independent-generator',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("sparse-generator", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-endpoint-carriers',
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("endpoint-carriers", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-history-actions',
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("history-actions", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-action-factorization',
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("factorization", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-source-firewall',
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("firewall", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-inference-controls',
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("inference-fixtures", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                'qualification-resource-dynamical-smooth',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-00", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-target-correlated',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-01", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-sink-multiscale',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-02", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-all-smooth',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-03", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-all-correlated',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-04", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-all-multiscale',
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("resource-canary-05", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                'qualification-resource-admission',
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("resource-admission", ReceiverHistoryCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                'qualification-closeout',
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("canary-close", ReceiverHistoryCanaryReport.SCHEMA),
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
            ('qualification-dense-equations', "dense-truth"),
            ('qualification-independent-generator', "sparse-generator"),
            ('qualification-endpoint-carriers', "endpoint-carriers"),
            ('qualification-history-actions', "history-actions"),
            ('qualification-action-factorization', "factorization"),
        ):
            canary_links_list.append(
                _Link(
                    producer,
                    output_id,
                    'qualification-source-firewall',
                    output_id,
                    ScientificInputRole.QUALIFICATION,
                )
            )
        resource_ids = (
            'qualification-resource-dynamical-smooth',
            'qualification-resource-target-correlated',
            'qualification-resource-sink-multiscale',
            'qualification-resource-all-smooth',
            'qualification-resource-all-correlated',
            'qualification-resource-all-multiscale',
        )
        for index, task_id in enumerate(resource_ids):
            canary_links_list.extend(
                (
                    _Link(
                        'qualification-source-firewall',
                        "firewall",
                        task_id,
                        "firewall-report",
                        ScientificInputRole.QUALIFICATION,
                    ),
                    _Link(
                        'qualification-inference-controls',
                        "inference-fixtures",
                        task_id,
                        "inference-report",
                        ScientificInputRole.QUALIFICATION,
                    ),
                    _Link(
                        task_id,
                        f"resource-canary-{index:02d}",
                        'qualification-resource-admission',
                        f"resource-canary-{index:02d}",
                        ScientificInputRole.QUALIFICATION,
                    ),
                )
            )
        canary_links_list.extend(
            (
                _Link(
                    'qualification-profile-conformance',
                    "profile-conformance",
                    'qualification-closeout',
                    "profile-conformance",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'qualification-source-firewall',
                    "firewall",
                    'qualification-closeout',
                    "firewall",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'qualification-inference-controls',
                    "inference-fixtures",
                    'qualification-closeout',
                    "inference-fixtures",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'qualification-resource-admission',
                    "resource-admission",
                    'qualification-closeout',
                    "resource-admission",
                    ScientificInputRole.QUALIFICATION,
                ),
            )
        )
        canary_links = tuple(canary_links_list)
        return canary_steps, canary_links
    if config.phase is ReceiverHistoryPhase.DEVELOPMENT:
        development_steps: list[_Step] = []
        development_links: list[_Link] = []
        for unit_id in config.unit_ids:
            tag = unit_id
            ids = {
                "descriptor": f"development-denominator.{tag}",
                "history": f"development-history.{tag}",
                "target": f"development-endpoint-geometry.{tag}",
                "freeze": f"development-challenge-freeze.{tag}",
                "generator": f"development-independent-generation-certificate.{tag}",
                "adjudicate": f"development-unit-adjudication.{tag}",
            }
            development_steps.extend(
                (
                    _Step(
                        ids["descriptor"],
                        ScientificStage.PREPARE,
                        DENOMINATOR_KEY,
                        (_Output("denominator-bundle", ReceiverHistoryDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", ReceiverHistoryHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", ReceiverHistoryObserverBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["freeze"],
                        ScientificStage.FREEZE,
                        NOMINATION_FREEZE_KEY,
                        (_Output("nomination-freeze", ReceiverHistoryNominationFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        DEVELOPMENT_GENERATOR_KEY,
                        _array_outputs("generator", ReceiverHistoryGeneratorBundle.SCHEMA),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.FALSIFY,
                        DEVELOPMENT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", ReceiverHistoryAdjudicationBundle.SCHEMA),),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                )
            )
            development_links.extend(_target_unit_links(ids))
        adjudicators = tuple(f"development-unit-adjudication.{unit_id}" for unit_id in config.unit_ids)
        development_steps.extend(
            (
                _Step(
                    'development-ledger',
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (_Output("development-ledger", ReceiverHistoryDevelopmentLedger.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    'development-integrity-resource-source-gate',
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (_Output("development-gate", ReceiverHistoryDevelopmentGate.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    'development-power-atlas',
                    ScientificStage.QUALIFY,
                    ENDPOINT_POWER_KEY,
                    (
                        _Output(
                            "endpoint-coordinate-power-atlas",
                            ReceiverHistoryEndpointCoordinatePowerAtlas.SCHEMA,
                        ),
                    ),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    'development-method-freeze',
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("method-freeze", ReceiverHistoryMethodFreeze.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    'development-design-freeze',
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (
                        _Output(
                            "evaluation-design-freeze", ReceiverHistoryEvaluationDesignFreeze.SCHEMA
                        ),
                    ),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    'development-closeout',
                    ScientificStage.REPORT,
                    DEVELOPMENT_REPORTER_KEY,
                    (
                        _Output("development-closeout", ReceiverHistoryPhaseCloseout.SCHEMA),
                        _Output(
                            "development-metatheory-gate",
                            ReceiverHistoryDevelopmentMetatheoryGate.SCHEMA,
                        ),
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
                    'development-ledger',
                    f"adjudication-{task_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        development_links.extend(
            (
                _Link(
                    'development-ledger',
                    "development-ledger",
                    'development-integrity-resource-source-gate',
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'development-integrity-resource-source-gate',
                    "development-gate",
                    'development-method-freeze',
                    "development-gate",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'development-ledger',
                    "development-ledger",
                    'development-power-atlas',
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'development-ledger',
                    "development-ledger",
                    'development-method-freeze',
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'development-method-freeze',
                    "method-freeze",
                    'development-design-freeze',
                    "method-freeze",
                    ScientificInputRole.MODEL,
                ),
                _Link(
                    'development-power-atlas',
                    "endpoint-coordinate-power-atlas",
                    'development-design-freeze',
                    "endpoint-coordinate-power-atlas",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'development-design-freeze',
                    "evaluation-design-freeze",
                    'development-closeout',
                    "evaluation-design-freeze",
                    ScientificInputRole.QUALIFICATION,
                ),
            )
        )
        return tuple(development_steps), tuple(development_links)
    if config.phase is ReceiverHistoryPhase.TARGETED_EVALUATION:
        steps: list[_Step] = []
        links: list[_Link] = []
        for unit_id in config.unit_ids:
            ids = {
                "descriptor": f"targeted-denominator.{unit_id}",
                "history": f"targeted-history.{unit_id}",
                "target": f"targeted-endpoint-geometry.{unit_id}",
                "freeze": f"targeted-challenge-freeze.{unit_id}",
                "generator": f"targeted-independent-generation-certificate.{unit_id}",
                "adjudicate": f"targeted-unit-adjudication.{unit_id}",
            }
            steps.extend(
                (
                    _Step(
                        ids["descriptor"],
                        ScientificStage.PREPARE,
                        DENOMINATOR_KEY,
                        (_Output("denominator-bundle", ReceiverHistoryDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", ReceiverHistoryHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", ReceiverHistoryObserverBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["freeze"],
                        ScientificStage.FREEZE,
                        NOMINATION_FREEZE_KEY,
                        (_Output("nomination-freeze", ReceiverHistoryNominationFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        GENERATOR_KEY,
                        _array_outputs("generator", ReceiverHistoryGeneratorBundle.SCHEMA),
                        OutcomeAccess.EVALUATION_SEALED,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.EVALUATE,
                        UNIT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", ReceiverHistoryAdjudicationBundle.SCHEMA),),
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
                    'phase-diagram-synthesis',
                    ScientificStage.SYNTHESIZE,
                    RECURRENCE_KEY,
                    (
                        _Output("bootstrap-summary", ReceiverHistoryBootstrapSummary.SCHEMA),
                        _Output("recurrence-result", ReceiverHistoryRecurrenceResult.SCHEMA),
                        _Output(
                            "inference-disposition-matrix",
                            ReceiverHistoryInferenceDispositionMatrix.SCHEMA,
                        ),
                    ),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SYNTHESIS,
                ),
                _Step(
                    'empirical-metatheory-closeout',
                    ScientificStage.REPORT,
                    REPORTER_KEY,
                    (
                        _Output(
                            "empirical-metatheory-dossier",
                            ReceiverHistoryEmpiricalMetatheoryDossier.SCHEMA,
                        ),
                        _Output(
                            "t-continuation-gate", ReceiverHistoryTargetedContinuationGateRecord.SCHEMA
                        ),
                        _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                        _Output("targeted-closeout", ReceiverHistoryTerminalCloseout.SCHEMA),
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
                    f"targeted-unit-adjudication.{unit_id}",
                    "adjudication-bundle",
                    'phase-diagram-synthesis',
                    f"adjudication-{unit_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        links.extend(
            (
                _Link(
                    'phase-diagram-synthesis',
                    "bootstrap-summary",
                    'empirical-metatheory-closeout',
                    "bootstrap-summary",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    'phase-diagram-synthesis',
                    "inference-disposition-matrix",
                    'empirical-metatheory-closeout',
                    "inference-disposition-matrix",
                    ScientificInputRole.OUTCOME,
                ),
                _Link(
                    'phase-diagram-synthesis',
                    "recurrence-result",
                    'empirical-metatheory-closeout',
                    "recurrence-result",
                    ScientificInputRole.OUTCOME,
                ),
            )
        )
        return tuple(steps), tuple(links)

    if config.phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
        steps = []
        links = []
        for unit_id in config.unit_ids:
            ids = {
                "untouched": f"untouched-preparation.{unit_id}",
                "freeze": f"untouched-freeze.{unit_id}",
                "generator": f"untouched-independent-generator.{unit_id}",
                "adjudicate": f"untouched-adjudication.{unit_id}",
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
                        (_Output("untouched-freeze", ReceiverHistoryUntouchedFreeze.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.FREEZE,
                        _SMALL,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        GENERATOR_KEY,
                        _array_outputs("generator", ReceiverHistoryGeneratorBundle.SCHEMA),
                        OutcomeAccess.EVALUATION_SEALED,
                        VisibilityCeiling.PROSPECTIVE,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.EVALUATE,
                        UNIT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", ReceiverHistoryAdjudicationBundle.SCHEMA),),
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
                    "untouched-phase-diagram-synthesis",
                    ScientificStage.SYNTHESIZE,
                    RECURRENCE_KEY,
                    (_Output("recurrence-result", ReceiverHistoryRecurrenceResult.SCHEMA),),
                    OutcomeAccess.EVALUATION_REVEALED,
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    BarrierKind.NONE,
                    _SYNTHESIS,
                ),
                _Step(
                    'integrated-phase-diagram-closeout',
                    ScientificStage.REPORT,
                    REPORTER_KEY,
                    (
                        _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                        _Output("terminal-closeout", ReceiverHistoryTerminalCloseout.SCHEMA),
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
                    f"untouched-adjudication.{unit_id}",
                    "adjudication-bundle",
                    "untouched-phase-diagram-synthesis",
                    f"adjudication-{unit_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        links.append(
            _Link(
                "untouched-phase-diagram-synthesis",
                "recurrence-result",
                'integrated-phase-diagram-closeout',
                "recurrence-result",
                ScientificInputRole.OUTCOME,
            )
        )
        return tuple(steps), tuple(links)
    raise ValueError("unsupported receiver-history phase")


def _untouched_outputs() -> tuple[_Output, ...]:
    return (
        _Output("denominator-bundle", ReceiverHistoryDenominatorBundle.SCHEMA),
        _Output("untouched-float-array-manifest", ReceiverHistoryArrayManifest.SCHEMA),
        _Output("untouched-float-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output("untouched-int-array-manifest", ReceiverHistoryArrayManifest.SCHEMA),
        _Output("untouched-int-arrays", INT_ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output("untouched-bundle", ReceiverHistoryUntouchedBundle.SCHEMA),
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


def protocol_template(
    *, registry: CapabilityRegistry, config: ReceiverHistoryConfig
) -> ProtocolTemplate:
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
                    "receiver-history-seed-roster-exclusive"
                    if definition.step_id.startswith("n")
                    else "receiver-history-terminal-exclusive"
                    if definition.step_id in {
                        "phase-diagram-synthesis",
                        "empirical-metatheory-closeout",
                        "untouched-phase-diagram-synthesis",
                        "integrated-phase-diagram-closeout",
                    }
                    else f"receiver-history-{definition.step_id}",
                ),
                barrier=definition.barrier,
                maximum_attempts=2,
                obligation_ids=(f"receiver-history-{definition.step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"receiver-history-{config.phase.value.lower()}-protocol",
        template_version=VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


@dataclass(frozen=True, slots=True)
class ReceiverHistoryExternalRecord:
    input_id: str
    record: CanonicalRecord
    role: ScientificInputRole
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling


def _validate_external_record_roster(
    config: ReceiverHistoryConfig,
    records: tuple[ReceiverHistoryExternalRecord, ...],
) -> None:
    expected_ids: tuple[str, ...] = {
        ReceiverHistoryPhase.NOMINATION: (),
        ReceiverHistoryPhase.CANARY: (),
        ReceiverHistoryPhase.DEVELOPMENT: (
            "input.receiver-history.development.canary-qualification",
            "input.receiver-history.development.evaluation-config",
            "input.receiver-history.development.seed-roster-commitment",
        ),
        ReceiverHistoryPhase.TARGETED_EVALUATION: (
            'input.receiver-history.targeted-evaluation.design-freeze',
            'input.receiver-history.targeted-evaluation.seed-roster',
        ),
        ReceiverHistoryPhase.UNTOUCHED_EVALUATION: (
            'input.receiver-history.untouched-evaluation.design-freeze',
            'input.receiver-history.untouched-evaluation.seed-roster',
            'input.receiver-history.untouched-evaluation.targeted-result',
            *tuple(
                f"input.receiver-history.untouched-evaluation.descriptor.{unit_id}"
                for unit_id in config.unit_ids
            ),
        ),
    }[config.phase]
    if tuple(value.input_id for value in records) != expected_ids:
        raise ValueError("receiver-history phase external record roster differs")
    if config.phase in {ReceiverHistoryPhase.NOMINATION, ReceiverHistoryPhase.CANARY}:
        return
    if config.phase is ReceiverHistoryPhase.DEVELOPMENT:
        canaries = tuple(
            value for value in records if isinstance(value.record, ReceiverHistoryCanaryReport)
        )
        configs = tuple(
            value for value in records if isinstance(value.record, ReceiverHistoryConfig)
        )
        commitments = tuple(
            value
            for value in records
            if isinstance(value.record, ReceiverHistorySeedRosterCommitment)
        )
        if len(canaries) != 1 or len(configs) != 1 or len(commitments) != 1:
            raise ValueError("receiver-history development parent record types differ")
        canary = canaries[0]
        commitment_record = commitments[0]
        evaluation_config_record = configs[0].record
        assert isinstance(evaluation_config_record, ReceiverHistoryConfig)
        if evaluation_config_record.phase is not ReceiverHistoryPhase.TARGETED_EVALUATION:
            raise ValueError("receiver-history evaluation config phase differs")
        assert isinstance(canary.record, ReceiverHistoryCanaryReport)
        assert isinstance(commitment_record.record, ReceiverHistorySeedRosterCommitment)
        if (
            not canary.record.passed
            or evaluation_config_record.seed_roster_commitment_sha256
            != commitment_record.record.fingerprint()
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
            raise ValueError("receiver-history development parent record contract differs")
        return
    designs = tuple(
        value
        for value in records
        if isinstance(value.record, ReceiverHistoryEvaluationDesignFreeze)
    )
    rosters = tuple(
        value for value in records if isinstance(value.record, ReceiverHistorySeedRoster)
    )
    if len(designs) != 1 or len(rosters) != 1:
        raise ValueError("receiver-history evaluation design/roster record types differ")
    design = designs[0]
    roster = rosters[0]
    assert isinstance(design.record, ReceiverHistoryEvaluationDesignFreeze)
    assert isinstance(roster.record, ReceiverHistorySeedRoster)
    seed_commitment = design.record.seed_roster_commitment
    roster_bytes = roster.record.canonical_bytes()
    if (
        design.record.evaluation_config_sha256 != config.fingerprint()
        or config.seed_roster_commitment_sha256 != seed_commitment.fingerprint()
        or tuple(value.unit_id for value in roster.record.entries) != config.unit_ids
        or sha256(roster_bytes).hexdigest() != seed_commitment.seed_payload_sha256
        or sha256(b'receiver-history-seed-roster-commitment\x00' + roster_bytes).hexdigest()
        != seed_commitment.commitment_sha256
        or (
            config.phase is ReceiverHistoryPhase.TARGETED_EVALUATION
            and not design.record.power_atlas.eligible_cell_ids
        )
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
        raise ValueError("receiver-history evaluation parent record contract differs")
    if config.phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
        targeted = tuple(
            value for value in records if isinstance(value.record, ReceiverHistoryRecurrenceResult)
        )
        descriptors = tuple(
            value for value in records if isinstance(value.record, ReceiverHistoryDenominatorBundle)
        )
        targeted_result = targeted[0].record if len(targeted) == 1 else None
        assert targeted_result is None or isinstance(
            targeted_result, ReceiverHistoryRecurrenceResult
        )
        denominator_records = tuple(value.record for value in descriptors)
        assert all(
            isinstance(value, ReceiverHistoryDenominatorBundle) for value in denominator_records
        )
        if (
            len(targeted) != 1
            or targeted_result is None
            or targeted_result.untouched_cells
            or len(descriptors) != 90
            or tuple(
                value.unit_id
                for value in denominator_records
                if isinstance(value, ReceiverHistoryDenominatorBundle)
            )
            != config.unit_ids
            or any(
                tuple(value.scale_cells for value in descriptor.descriptors) != config.scale_cells
                for descriptor in denominator_records
                if isinstance(descriptor, ReceiverHistoryDenominatorBundle)
            )
            or any(
                value.preparation_substream_seed_sha256 == "0" * 64
                for value in denominator_records
                if isinstance(value, ReceiverHistoryDenominatorBundle)
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
            raise ValueError('receiver-history conditional untouched external records differ')


def scientific_graph(
    *,
    registry: CapabilityRegistry,
    config: ReceiverHistoryConfig,
    protocol: ProtocolTemplate,
    external_records: tuple[ReceiverHistoryExternalRecord, ...] = (),
) -> CandidateScientificGraph:
    _validate_external_record_roster(config, external_records)
    definitions, links = _phase_definitions(config)
    if {value.step_id for value in definitions} != {value.step_id for value in protocol.steps}:
        raise ValueError("receiver-history protocol/phase definitions differ")
    steps = {value.step_id: value for value in protocol.steps}
    outputs = {
        (step.step_id, output.output_id): output
        for step in protocol.steps
        for output in step.outputs
    }
    config_external = ReceiverHistoryExternalRecord(
        input_id=f"input.receiver-history.{config.phase.value.lower()}.config",
        record=config,
        role=ScientificInputRole.MODEL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    records = tuple(sorted((config_external, *external_records), key=lambda value: value.input_id))
    input_ids = tuple(value.input_id for value in records)
    if input_ids != tuple(sorted(set(input_ids))):
        raise ValueError("receiver-history external records must be sorted and unique")
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
        graph_id=f"graph.receiver-history.{config.phase.value.lower()}",
        external_inputs=external_inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _external_edge(
    record: ReceiverHistoryExternalRecord,
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
    phase: ReceiverHistoryPhase,
    record: CanonicalRecord,
    protocol: ProtocolTemplate,
) -> tuple[tuple[str, str], ...]:
    task_ids = {value.step_id for value in protocol.steps}
    if phase is ReceiverHistoryPhase.DEVELOPMENT:
        if isinstance(record, ReceiverHistoryCanaryReport):
            return (('development-integrity-resource-source-gate', "canary-qualification"),)
        if isinstance(record, ReceiverHistorySeedRosterCommitment):
            return (('development-design-freeze', "seed-roster-commitment"),)
        if (
            isinstance(record, ReceiverHistoryConfig)
            and record.phase is ReceiverHistoryPhase.TARGETED_EVALUATION
        ):
            return (('development-design-freeze', "evaluation-config"),)
    if phase is ReceiverHistoryPhase.TARGETED_EVALUATION:
        if isinstance(record, ReceiverHistorySeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith('targeted-denominator.')
            )
        if isinstance(record, ReceiverHistoryEvaluationDesignFreeze):
            consumers = tuple(
                task_id
                for task_id in sorted(task_ids)
                if task_id.startswith(
                    (
                        'targeted-denominator.',
                        'targeted-history.',
                        'targeted-endpoint-geometry.',
                        'targeted-challenge-freeze.',
                        'targeted-independent-generation-certificate.',
                        'targeted-unit-adjudication.',
                    )
                )
                or task_id in {'phase-diagram-synthesis', 'empirical-metatheory-closeout'}
            )
            return tuple((value, "evaluation-design-freeze") for value in consumers)
    if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
        if isinstance(record, ReceiverHistoryEvaluationDesignFreeze):
            return tuple((task_id, "evaluation-design-freeze") for task_id in sorted(task_ids))
        if isinstance(record, ReceiverHistorySeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith('untouched-preparation.')
            )
        if isinstance(record, ReceiverHistoryDenominatorBundle):
            suffix = record.unit_id
            return ((f"untouched-preparation.{suffix}", "denominator-bundle"),)
        if isinstance(record, ReceiverHistoryRecurrenceResult):
            return (("untouched-phase-diagram-synthesis", "targeted-result"),)
    raise ValueError("receiver-history external record has no exact phase consumer roster")


def study_template(
    *,
    registry: CapabilityRegistry,
    config: ReceiverHistoryConfig,
    protocol: ProtocolTemplate,
    experiment: ExperimentSpec,
    external_records: tuple[ReceiverHistoryExternalRecord, ...] = (),
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
        raise ValueError("receiver-history phase must have exactly one terminal reporter")
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
        template_key=f"receiver-history.{config.phase.value.lower()}",
        template_version=VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.receiver-history.{config.phase.value.lower()}",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


def candidate_catalog(
    *, templates: tuple[StudyTemplate, ...], registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id="catalog.receiver-history-simulator-morphism-challenges",
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
class ReceiverHistoryConfigDecoder:
    provider_key: str
    config: ReceiverHistoryConfig
    provider_version: str = VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != ReceiverHistoryConfig.SCHEMA:
            raise ValueError("receiver-history config schema differs")
        if decode_config(payload) != self.config:
            raise ValueError("receiver-history config differs from the exact phase registration")


def config_decoders(
    catalog: CandidateCapabilityCatalog,
    config: ReceiverHistoryConfig,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return tuple(
        ReceiverHistoryConfigDecoder(value.provider_key, config) for value in catalog.registrations
    )


__all__ = [
    "ARRAY_PAYLOAD_SCHEMA",
    "CONFIG_MEDIA_TYPE",
    "DEVELOPMENT_ADJUDICATOR_KEY",
    "DEVELOPMENT_GENERATOR_KEY",
    "ENDPOINT_POWER_KEY",
    "DENOMINATOR_KEY",
    "GENERATOR_KEY",
    "HISTORY_KEY",
    "ReceiverHistoryConfigDecoder",
    "ReceiverHistoryExternalRecord",
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
    "receiver_history_registry",
    "receiver_history_phase_registry",
    'study_template',
    "protocol_template",
    "scientific_graph",
]
