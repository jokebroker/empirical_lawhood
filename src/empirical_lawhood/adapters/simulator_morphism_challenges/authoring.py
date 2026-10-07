"""Static capabilities, protocols, and exact scientific DAGs for simulator morphism challenges."""

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

from .contracts import SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengePhase, SimulatorMorphismChallengeRecurrenceResult, SimulatorMorphismChallengeSeedRosterCommitment, SimulatorMorphismChallengeTerminalCloseout
from .runtime_contracts import SimulatorMorphismChallengeAdjudicationBundle, SimulatorMorphismChallengeArrayManifest, SimulatorMorphismChallengeBootstrapSummary, SimulatorMorphismChallengeCanaryReport, SimulatorMorphismChallengeDenominatorBundle, SimulatorMorphismChallengeDevelopmentGate, SimulatorMorphismChallengeDevelopmentLedger, SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeGeneratorBundle, SimulatorMorphismChallengeHistoryBundle, SimulatorMorphismChallengeNominationFreeze, SimulatorMorphismChallengeObserverBundle, SimulatorMorphismChallengePhaseCloseout, SimulatorMorphismChallengeRequestedUnitLedger, SimulatorMorphismChallengeSeedRoster


VERSION = "1.0.0"
ARRAY_PAYLOAD_SCHEMA = 'empirical-lawhood/simulator-morphism-challenges/array-payload-npy'
CONFIG_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NPY_MEDIA_TYPE = "application/x-npy"


def decode_config(payload: bytes) -> SimulatorMorphismChallengeConfig:
    # Static discovery must not import the NumPy descriptor producer.
    from .descriptors import decode_config as decode

    return decode(payload)

SEED_ROSTER_KEY = "simulator-morphism-challenges.seed-roster"
DENOMINATOR_KEY = "simulator-morphism-challenges.denominator-descriptor"
GENERATOR_KEY = "simulator-morphism-challenges.independent-generator"
DEVELOPMENT_GENERATOR_KEY = "simulator-morphism-challenges.development-generator"
HISTORY_KEY = "simulator-morphism-challenges.history-observer"
TARGETER_KEY = "simulator-morphism-challenges.boundary-targeter"
NOMINATION_FREEZE_KEY = "simulator-morphism-challenges.nomination-freeze"
UNIT_ADJUDICATOR_KEY = "simulator-morphism-challenges.unit-adjudicator"
DEVELOPMENT_ADJUDICATOR_KEY = "simulator-morphism-challenges.development-adjudicator"
RECURRENCE_KEY = "simulator-morphism-challenges.recurrence-synthesizer"
REPORTER_KEY = "simulator-morphism-challenges.reporter"
DEVELOPMENT_REPORTER_KEY = "simulator-morphism-challenges.development-reporter"
METHOD_FREEZE_KEY = "simulator-morphism-challenges.method-freeze"
DESIGN_QUALIFIER_KEY = "simulator-morphism-challenges.design-qualifier"
DEVELOPMENT_QUALIFIER_KEY = "simulator-morphism-challenges.development-qualifier"

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
    wall_time_seconds=300,
    source_scan_bytes=32 * 1024**2,
    output_bytes=32 * 1024**2,
)
_OBSERVER = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=900,
    source_scan_bytes=32 * 1024**2,
    output_bytes=128 * 1024**2,
)
_GENERATOR = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=1800,
    source_scan_bytes=32 * 1024**2,
    output_bytes=128 * 1024**2,
)
_SYNTHESIS = ResourceBudget(
    cpu_cores=2,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=600,
    source_scan_bytes=32 * 1024**2,
    output_bytes=128 * 1024**2,
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
        (SimulatorMorphismChallengeConfig.SCHEMA,),
        (SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA,),
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
                    SimulatorMorphismChallengeAdjudicationBundle.SCHEMA,
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentGate.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentLedger.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeHistoryBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, SimulatorMorphismChallengeArrayManifest.SCHEMA, SimulatorMorphismChallengeObserverBundle.SCHEMA)
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _OBSERVER,
    ),
    _CapabilityDefinition(
        DENOMINATOR_KEY,
        CapabilityKind.SIMULATOR,
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeSeedRoster.SCHEMA,
                )
            )
        ),
        (SimulatorMorphismChallengeDenominatorBundle.SCHEMA,),
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
                    SimulatorMorphismChallengeArrayManifest.SCHEMA,
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeGeneratorBundle.SCHEMA,
                    SimulatorMorphismChallengeObserverBundle.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                )
            )
        ),
        tuple(sorted((SimulatorMorphismChallengeAdjudicationBundle.SCHEMA, SimulatorMorphismChallengeCanaryReport.SCHEMA))),
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeNominationFreeze.SCHEMA,
                    SimulatorMorphismChallengeObserverBundle.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SimulatorMorphismChallengeArrayManifest.SCHEMA,
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeGeneratorBundle.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeNominationFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (ARRAY_PAYLOAD_SCHEMA, SimulatorMorphismChallengeArrayManifest.SCHEMA, SimulatorMorphismChallengeGeneratorBundle.SCHEMA)
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    ARRAY_PAYLOAD_SCHEMA,
                    SimulatorMorphismChallengeArrayManifest.SCHEMA,
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeHistoryBundle.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeHistoryBundle.SCHEMA,
                    SimulatorMorphismChallengeObserverBundle.SCHEMA,
                    SimulatorMorphismChallengeSeedRoster.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeNominationFreeze.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                )
            )
        ),
        _DEVELOPMENT,
        OutcomeAccess.OUTCOME_BLIND,
        _SMALL,
    ),
    _CapabilityDefinition(
        METHOD_FREEZE_KEY,
        CapabilityKind.TRANSFORM,
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentGate.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentLedger.SCHEMA,
                    SimulatorMorphismChallengeMethodFreeze.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(sorted((SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA, SimulatorMorphismChallengeMethodFreeze.SCHEMA))),
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
                    SimulatorMorphismChallengeAdjudicationBundle.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeMethodFreeze.SCHEMA,
                )
            )
        ),
        tuple(sorted((SimulatorMorphismChallengeBootstrapSummary.SCHEMA, SimulatorMorphismChallengeRecurrenceResult.SCHEMA))),
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
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeBootstrapSummary.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                    SimulatorMorphismChallengeRecurrenceResult.SCHEMA,
                    SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentGate.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentLedger.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                    SimulatorMorphismChallengeTerminalCloseout.SCHEMA,
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
                    SimulatorMorphismChallengeAdjudicationBundle.SCHEMA,
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentGate.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengeCanaryReport.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentGate.SCHEMA,
                    SimulatorMorphismChallengeDevelopmentLedger.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
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
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                    SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA,
                    SimulatorMorphismChallengeSeedRoster.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
                )
            )
        ),
        tuple(
            sorted(
                (
                    SimulatorMorphismChallengePhaseCloseout.SCHEMA,
                    SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA,
                    SimulatorMorphismChallengeSeedRoster.SCHEMA,
                    SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA,
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
                    SimulatorMorphismChallengeArrayManifest.SCHEMA,
                    SimulatorMorphismChallengeConfig.SCHEMA,
                    SimulatorMorphismChallengeDenominatorBundle.SCHEMA,
                    SimulatorMorphismChallengeGeneratorBundle.SCHEMA,
                    SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA,
                    SimulatorMorphismChallengeMethodFreeze.SCHEMA,
                    SimulatorMorphismChallengeNominationFreeze.SCHEMA,
                )
            )
        ),
        (SimulatorMorphismChallengeAdjudicationBundle.SCHEMA,),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
        _SMALL,
    ),
)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("ascii")).hexdigest()


def simulator_morphism_challenges_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    from .capability_configs import CAPABILITY_CONFIG_TYPES

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities = tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=value.key,
                    capability_version=VERSION,
                    kind=value.kind,
                    config_schema=CAPABILITY_CONFIG_TYPES[value.key].SCHEMA,
                    config_schema_sha256=_schema_sha256(CAPABILITY_CONFIG_TYPES[value.key].SCHEMA),
                    input_schema_ids=value.inputs,
                    output_schema_ids=value.outputs,
                    permissions=value.permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=value.maximum_access,
                    resource_ceiling=value.budget,
                    deterministic=value.deterministic,
                    seed_required=False,
                    language_id="python",
                    runtime_id="cpython-numpy-scipy-simulator-morphism-challenges",
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
        registry_id="simulator-morphism-challenges-simulator-morphism-registry",
        capabilities=capabilities,
    )


def simulator_morphism_challenges_phase_registry(
    *,
    implementation_sha256: str,
    config: SimulatorMorphismChallengeConfig,
) -> CapabilityRegistry:
    """Return the least-privilege registry frozen into one phase candidate."""

    complete = simulator_morphism_challenges_registry(implementation_sha256=implementation_sha256)
    definitions, _links = _phase_definitions(config)
    selected_keys = {value.key for value in definitions}
    capabilities = tuple(
        value for value in complete.capabilities if value.capability_key in selected_keys
    )
    if {value.capability_key for value in capabilities} != selected_keys:
        raise ValueError("simulator morphism challenges phase registry lacks a selected capability")
    return CapabilityRegistry(
        registry_id=f"simulator-morphism-challenges-{config.phase.value.lower()}-registry",
        capabilities=capabilities,
    )


def config_ref(config: SimulatorMorphismChallengeConfig, capability_key: str) -> CapabilityConfigRef:
    from .capability_configs import CAPABILITY_CONFIG_TYPES

    config = CAPABILITY_CONFIG_TYPES[capability_key](config)
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
        _Output(f"{prefix}-array-manifest", SimulatorMorphismChallengeArrayManifest.SCHEMA),
        _Output(f"{prefix}-arrays", ARRAY_PAYLOAD_SCHEMA, ArtifactProfile.NUMPY_NO_PICKLE),
        _Output(f"{prefix}-bundle", bundle_schema),
    )


def _phase_definitions(config: SimulatorMorphismChallengeConfig) -> tuple[tuple[_Step, ...], tuple[_Link, ...]]:
    if config.phase is SimulatorMorphismChallengePhase.NOMINATION:
        nomination_steps = (
            _Step(
                "nomination-validate-design",
                ScientificStage.QUALIFY,
                DESIGN_QUALIFIER_KEY,
                (_Output("requested-unit-ledger", SimulatorMorphismChallengeRequestedUnitLedger.SCHEMA),),
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
                    _Output("roster-commitment", SimulatorMorphismChallengeSeedRosterCommitment.SCHEMA),
                    _Output("seed-roster", SimulatorMorphismChallengeSeedRoster.SCHEMA),
                ),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "nomination-seal-roster",
                ScientificStage.FREEZE,
                NOMINATION_FREEZE_KEY,
                (_Output("roster-freeze-closeout", SimulatorMorphismChallengePhaseCloseout.SCHEMA),),
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
                    _Output("nomination-closeout", SimulatorMorphismChallengePhaseCloseout.SCHEMA),
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
    if config.phase is SimulatorMorphismChallengePhase.CANARY:
        canary_steps = (
            _Step(
                "canary-contract-conformance",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("contract-report", SimulatorMorphismChallengePhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "canary-generator",
                ScientificStage.ACQUIRE,
                DEVELOPMENT_GENERATOR_KEY,
                (_Output("generator-canary", SimulatorMorphismChallengeCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "canary-observer",
                ScientificStage.DEVELOP,
                HISTORY_KEY,
                (_Output("observer-canary", SimulatorMorphismChallengeCanaryReport.SCHEMA),),
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-independence-audit",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("independence-report", SimulatorMorphismChallengePhaseCloseout.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
            _Step(
                "canary-cross-implementation-conformance",
                ScientificStage.QUALIFY,
                DEVELOPMENT_ADJUDICATOR_KEY,
                (_Output("numerical-conformance", SimulatorMorphismChallengeCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _OBSERVER,
            ),
            _Step(
                "canary-resource",
                ScientificStage.QUALIFY,
                DEVELOPMENT_QUALIFIER_KEY,
                (_Output("resource-canary", SimulatorMorphismChallengeCanaryReport.SCHEMA),),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _GENERATOR,
            ),
            _Step(
                "canary-adjudicate",
                ScientificStage.REPORT,
                DEVELOPMENT_REPORTER_KEY,
                (
                    _Output("canary-qualification", SimulatorMorphismChallengeCanaryReport.SCHEMA),
                    _Output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                ),
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                BarrierKind.NONE,
                _SMALL,
            ),
        )
        canary_links = (
            _Link(
                "canary-contract-conformance",
                "contract-report",
                "canary-resource",
                "contract-report",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "canary-generator",
                "generator-canary",
                "canary-cross-implementation-conformance",
                "generator-canary",
                ScientificInputRole.OUTCOME,
            ),
            _Link(
                "canary-observer",
                "observer-canary",
                "canary-cross-implementation-conformance",
                "observer-canary",
                ScientificInputRole.MODEL,
            ),
            _Link(
                "canary-independence-audit",
                "independence-report",
                "canary-cross-implementation-conformance",
                "independence-report",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "canary-cross-implementation-conformance",
                "numerical-conformance",
                "canary-adjudicate",
                "numerical-conformance",
                ScientificInputRole.QUALIFICATION,
            ),
            _Link(
                "canary-resource",
                "resource-canary",
                "canary-adjudicate",
                "resource-canary",
                ScientificInputRole.QUALIFICATION,
            ),
        )
        return canary_steps, canary_links
    if config.phase is SimulatorMorphismChallengePhase.DEVELOPMENT:
        development_steps: list[_Step] = []
        development_links: list[_Link] = []
        for unit_id in config.unit_ids:
            tag = unit_id
            ids = {
                "descriptor": f"development-descriptor.{tag}",
                "history": f"development-history.{tag}",
                "target": f"development-targeter.{tag}",
                "generator": f"development-generator.{tag}",
                "adjudicate": f"development-adjudicate.{tag}",
            }
            development_steps.extend(
                (
                    _Step(
                        ids["descriptor"],
                        ScientificStage.PREPARE,
                        DENOMINATOR_KEY,
                        (_Output("denominator-bundle", SimulatorMorphismChallengeDenominatorBundle.SCHEMA),),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _SMALL,
                    ),
                    _Step(
                        ids["history"],
                        ScientificStage.DEVELOP,
                        HISTORY_KEY,
                        _array_outputs("history", SimulatorMorphismChallengeHistoryBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["target"],
                        ScientificStage.FALSIFY,
                        TARGETER_KEY,
                        _array_outputs("targeter", SimulatorMorphismChallengeObserverBundle.SCHEMA),
                        OutcomeAccess.OUTCOME_BLIND,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _OBSERVER,
                    ),
                    _Step(
                        ids["generator"],
                        ScientificStage.ACQUIRE,
                        DEVELOPMENT_GENERATOR_KEY,
                        _array_outputs("generator", SimulatorMorphismChallengeGeneratorBundle.SCHEMA),
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        VisibilityCeiling.DEVELOPMENT_ONLY,
                        BarrierKind.NONE,
                        _GENERATOR,
                    ),
                    _Step(
                        ids["adjudicate"],
                        ScientificStage.FALSIFY,
                        DEVELOPMENT_ADJUDICATOR_KEY,
                        (_Output("adjudication-bundle", SimulatorMorphismChallengeAdjudicationBundle.SCHEMA),),
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
                    "development-power-and-correctness-gate",
                    ScientificStage.QUALIFY,
                    DEVELOPMENT_QUALIFIER_KEY,
                    (
                        _Output("development-ledger", SimulatorMorphismChallengeDevelopmentLedger.SCHEMA),
                        _Output("development-gate", SimulatorMorphismChallengeDevelopmentGate.SCHEMA),
                    ),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    "f0-freeze-method",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("method-freeze", SimulatorMorphismChallengeMethodFreeze.SCHEMA),),
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    "f1-freeze-evaluation-design",
                    ScientificStage.FREEZE,
                    METHOD_FREEZE_KEY,
                    (_Output("evaluation-design-freeze", SimulatorMorphismChallengeEvaluationDesignFreeze.SCHEMA),),
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
                        _Output("development-closeout", SimulatorMorphismChallengePhaseCloseout.SCHEMA),
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
                    "development-power-and-correctness-gate",
                    f"adjudication-{task_id}",
                    ScientificInputRole.OUTCOME,
                )
            )
        development_links.extend(
            (
                _Link(
                    "development-power-and-correctness-gate",
                    "development-ledger",
                    "f0-freeze-method",
                    "development-ledger",
                    ScientificInputRole.QUALIFICATION,
                ),
                _Link(
                    "development-power-and-correctness-gate",
                    "development-gate",
                    "f0-freeze-method",
                    "development-gate",
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
            "freeze": f"evaluation-freeze-nomination.{tag}",
            "generator": f"evaluation-generator.{tag}",
            "adjudicate": f"evaluation-adjudicate.{tag}",
        }
        evaluation_steps.extend(
            (
                _Step(
                    ids["descriptor"],
                    ScientificStage.PREPARE,
                    DENOMINATOR_KEY,
                    (_Output("denominator-bundle", SimulatorMorphismChallengeDenominatorBundle.SCHEMA),),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _SMALL,
                ),
                _Step(
                    ids["history"],
                    ScientificStage.DEVELOP,
                    HISTORY_KEY,
                    _array_outputs("history", SimulatorMorphismChallengeHistoryBundle.SCHEMA),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _OBSERVER,
                ),
                _Step(
                    ids["target"],
                    ScientificStage.FALSIFY,
                    TARGETER_KEY,
                    _array_outputs("targeter", SimulatorMorphismChallengeObserverBundle.SCHEMA),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _OBSERVER,
                ),
                _Step(
                    ids["freeze"],
                    ScientificStage.FREEZE,
                    NOMINATION_FREEZE_KEY,
                    (_Output("nomination-freeze", SimulatorMorphismChallengeNominationFreeze.SCHEMA),),
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.FREEZE,
                    _SMALL,
                ),
                _Step(
                    ids["generator"],
                    ScientificStage.ACQUIRE,
                    GENERATOR_KEY,
                    _array_outputs("generator", SimulatorMorphismChallengeGeneratorBundle.SCHEMA),
                    OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    BarrierKind.NONE,
                    _GENERATOR,
                ),
                _Step(
                    ids["adjudicate"],
                    ScientificStage.EVALUATE,
                    UNIT_ADJUDICATOR_KEY,
                    (_Output("adjudication-bundle", SimulatorMorphismChallengeAdjudicationBundle.SCHEMA),),
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
                    _Output("bootstrap-summary", SimulatorMorphismChallengeBootstrapSummary.SCHEMA),
                    _Output("recurrence-result", SimulatorMorphismChallengeRecurrenceResult.SCHEMA),
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
                    _Output("terminal-closeout", SimulatorMorphismChallengeTerminalCloseout.SCHEMA),
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
    values = [
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
    ]
    if evaluation:
        values.extend(
            (
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
                    ids["freeze"],
                    "nomination-freeze",
                    ids["adjudicate"],
                    "nomination-freeze",
                    ScientificInputRole.MODEL,
                ),
            )
        )
    else:
        values.extend(
            (
                _Link(
                    ids["descriptor"],
                    "denominator-bundle",
                    ids["generator"],
                    "denominator-bundle",
                    ScientificInputRole.DENOMINATOR,
                ),
                _Link(
                    ids["target"],
                    "targeter-bundle",
                    ids["generator"],
                    "observer-bundle",
                    ScientificInputRole.MODEL,
                ),
            )
        )
    values.extend(
        (
            _Link(
                ids["descriptor"],
                "denominator-bundle",
                ids["adjudicate"],
                "denominator-bundle",
                ScientificInputRole.DENOMINATOR,
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
    )
    if not evaluation:
        values.append(
            _Link(
                ids["target"],
                "targeter-bundle",
                ids["adjudicate"],
                "observer-bundle",
                ScientificInputRole.MODEL,
            )
        )
    return tuple(values)


def protocol_template(*, registry: CapabilityRegistry, config: SimulatorMorphismChallengeConfig) -> ProtocolTemplate:
    definitions, links = _phase_definitions(config)
    parents = {
        step.step_id: tuple(
            sorted({value.producer for value in links if value.consumer == step.step_id})
        )
        for step in definitions
    }
    steps = []
    for definition in definitions:
        manifest = registry.resolve(definition.key, VERSION)
        steps.append(
            ProtocolStepTemplate(
                step_id=definition.step_id,
                stage=definition.stage,
                capability_key=definition.key,
                capability_version=VERSION,
                config=config_ref(config, definition.key),
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
                    "simulator-morphism-challenges-seed-roster-exclusive"
                    if definition.step_id.startswith("n")
                    else "simulator-morphism-challenges-terminal-exclusive"
                    if definition.step_id.startswith("x")
                    else f"simulator-morphism-challenges-{definition.step_id}",
                ),
                barrier=definition.barrier,
                maximum_attempts=2,
                obligation_ids=(f"simulator-morphism-challenges-{definition.step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id=f"simulator-morphism-challenges-{config.phase.value.lower()}-protocol",
        template_version=VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeExternalRecord:
    input_id: str
    record: CanonicalRecord
    role: ScientificInputRole
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling


def _validate_external_record_roster(
    config: SimulatorMorphismChallengeConfig,
    records: tuple[SimulatorMorphismChallengeExternalRecord, ...],
) -> None:
    expected_ids = {
        SimulatorMorphismChallengePhase.NOMINATION: (),
        SimulatorMorphismChallengePhase.CANARY: (),
        SimulatorMorphismChallengePhase.DEVELOPMENT: (
            "input.simulator-morphism-challenges.development.canary-qualification",
            "input.simulator-morphism-challenges.development.evaluation-config",
            "input.simulator-morphism-challenges.development.seed-roster-commitment",
        ),
        SimulatorMorphismChallengePhase.EVALUATION: (
            "input.simulator-morphism-challenges.evaluation.design-freeze",
            "input.simulator-morphism-challenges.evaluation.seed-roster",
        ),
    }[config.phase]
    if tuple(value.input_id for value in records) != expected_ids:
        raise ValueError("simulator morphism challenges phase external record roster differs")
    if config.phase in {SimulatorMorphismChallengePhase.NOMINATION, SimulatorMorphismChallengePhase.CANARY}:
        return
    by_type = {type(value.record): value for value in records}
    if len(by_type) != len(records):
        raise ValueError("simulator morphism challenges phase external record types repeat")
    if config.phase is SimulatorMorphismChallengePhase.DEVELOPMENT:
        if set(by_type) != {
            SimulatorMorphismChallengeCanaryReport,
            SimulatorMorphismChallengeConfig,
            SimulatorMorphismChallengeSeedRosterCommitment,
        }:
            raise ValueError("simulator morphism challenges development parent record types differ")
        canary = by_type[SimulatorMorphismChallengeCanaryReport]
        evaluation_config = by_type[SimulatorMorphismChallengeConfig]
        commitment_record = by_type[SimulatorMorphismChallengeSeedRosterCommitment]
        assert isinstance(canary.record, SimulatorMorphismChallengeCanaryReport)
        assert isinstance(evaluation_config.record, SimulatorMorphismChallengeConfig)
        assert isinstance(commitment_record.record, SimulatorMorphismChallengeSeedRosterCommitment)
        if (
            not canary.record.passed
            or evaluation_config.record.phase is not SimulatorMorphismChallengePhase.EVALUATION
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
            raise ValueError("simulator morphism challenges development parent record contract differs")
        return
    if set(by_type) != {SimulatorMorphismChallengeEvaluationDesignFreeze, SimulatorMorphismChallengeSeedRoster}:
        raise ValueError("simulator morphism challenges evaluation parent record types differ")
    design = by_type[SimulatorMorphismChallengeEvaluationDesignFreeze]
    roster = by_type[SimulatorMorphismChallengeSeedRoster]
    assert isinstance(design.record, SimulatorMorphismChallengeEvaluationDesignFreeze)
    assert isinstance(roster.record, SimulatorMorphismChallengeSeedRoster)
    seed_commitment = design.record.seed_roster_commitment
    roster_bytes = roster.record.canonical_bytes()
    if (
        design.record.evaluation_config_sha256 != config.fingerprint()
        or config.seed_roster_commitment_sha256 != seed_commitment.fingerprint()
        or tuple(value.unit_id for value in roster.record.entries) != config.unit_ids
        or sha256(roster_bytes).hexdigest() != seed_commitment.seed_payload_sha256
        or sha256(b"simulator-morphism-challenges-seed-roster-commitment\0" + roster_bytes).hexdigest()
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
        raise ValueError("simulator morphism challenges evaluation parent record contract differs")


def scientific_graph(
    *,
    registry: CapabilityRegistry,
    config: SimulatorMorphismChallengeConfig,
    protocol: ProtocolTemplate,
    external_records: tuple[SimulatorMorphismChallengeExternalRecord, ...] = (),
) -> CandidateScientificGraph:
    _validate_external_record_roster(config, external_records)
    definitions, links = _phase_definitions(config)
    if {value.step_id for value in definitions} != {value.step_id for value in protocol.steps}:
        raise ValueError("simulator morphism challenges protocol/phase definitions differ")
    steps = {value.step_id: value for value in protocol.steps}
    outputs = {
        (step.step_id, output.output_id): output
        for step in protocol.steps
        for output in step.outputs
    }
    config_external = SimulatorMorphismChallengeExternalRecord(
        input_id=f"input.simulator-morphism-challenges.{config.phase.value.lower()}.config",
        record=config,
        role=ScientificInputRole.MODEL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    records = tuple(sorted((config_external, *external_records), key=lambda value: value.input_id))
    input_ids = tuple(value.input_id for value in records)
    if input_ids != tuple(sorted(set(input_ids))):
        raise ValueError("simulator morphism challenges external records must be sorted and unique")
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
        graph_id=f"graph.simulator-morphism-challenges.{config.phase.value.lower()}",
        external_inputs=external_inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _external_edge(
    record: SimulatorMorphismChallengeExternalRecord,
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
    phase: SimulatorMorphismChallengePhase,
    record: CanonicalRecord,
    protocol: ProtocolTemplate,
) -> tuple[tuple[str, str], ...]:
    task_ids = {value.step_id for value in protocol.steps}
    if phase is SimulatorMorphismChallengePhase.DEVELOPMENT:
        if isinstance(record, SimulatorMorphismChallengeCanaryReport):
            return (("development-power-and-correctness-gate", "canary-qualification"),)
        if isinstance(record, SimulatorMorphismChallengeSeedRosterCommitment):
            return (
                ("f0-freeze-method", "seed-roster-commitment"),
                ("f1-freeze-evaluation-design", "seed-roster-commitment"),
            )
        if isinstance(record, SimulatorMorphismChallengeConfig) and record.phase is SimulatorMorphismChallengePhase.EVALUATION:
            return (
                ("f0-freeze-method", "evaluation-config"),
                ("f1-freeze-evaluation-design", "evaluation-config"),
            )
    if phase is SimulatorMorphismChallengePhase.EVALUATION:
        if isinstance(record, SimulatorMorphismChallengeSeedRoster):
            return tuple(
                (task_id, "evaluation-seed-roster")
                for task_id in sorted(task_ids)
                if task_id.startswith("evaluation-descriptor.")
            )
        if isinstance(record, SimulatorMorphismChallengeEvaluationDesignFreeze):
            consumers = tuple(
                task_id
                for task_id in sorted(task_ids)
                if task_id.startswith(
                    (
                        "evaluation-descriptor.",
                        "evaluation-history.",
                        "evaluation-targeter.",
                        "evaluation-freeze-nomination.",
                        "evaluation-generator.",
                        "evaluation-adjudicate.",
                    )
                )
                or task_id in {"recurrence-synthesize", "terminal-report"}
            )
            return tuple((value, "evaluation-design-freeze") for value in consumers)
    raise ValueError("simulator morphism challenges external record has no exact phase consumer roster")


def study_template(
    *,
    registry: CapabilityRegistry,
    config: SimulatorMorphismChallengeConfig,
    protocol: ProtocolTemplate,
    experiment: ExperimentSpec,
    external_records: tuple[SimulatorMorphismChallengeExternalRecord, ...] = (),
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
    terminal = protocol.steps[-1]
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
        template_key=f"simulator-morphism-challenges.{config.phase.value.lower()}",
        template_version=VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.simulator-morphism-challenges.{config.phase.value.lower()}",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


def candidate_catalog(
    *, templates: tuple[StudyTemplate, ...], registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    return CandidateCapabilityCatalog(
        catalog_id="catalog.simulator-morphism-challenges-simulator-morphism-challenges",
        registrations=tuple(
            CandidateCapabilityRegistration(
                manifest=value,
                provider_key=f"{value.capability_key}.provider",
                provider_version=value.capability_version,
                config_media_type=CONFIG_MEDIA_TYPE,
                maximum_config_bytes=8 * 1024**2,
            )
            for value in registry.capabilities
        ),
        templates=tuple(sorted(templates, key=lambda value: value.template_key)),
    )


@dataclass(frozen=True, slots=True)
class SimulatorMorphismChallengeConfigDecoder:
    provider_key: str
    config: SimulatorMorphismChallengeConfig
    capability_key: str
    provider_version: str = VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        from .capability_configs import CAPABILITY_CONFIG_TYPES
        from empirical_lawhood.kernel.decoding import decode_canonical_bytes

        record_type = CAPABILITY_CONFIG_TYPES[self.capability_key]
        if expected_schema != record_type.SCHEMA:
            raise ValueError("simulator morphism challenges config schema differs")
        if decode_canonical_bytes(payload, record_type, maximum_bytes=512 * 1024).config != self.config:
            raise ValueError("simulator morphism challenges config differs from the exact phase registration")


def config_decoders(
    catalog: CandidateCapabilityCatalog,
    config: SimulatorMorphismChallengeConfig,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return tuple(
        SimulatorMorphismChallengeConfigDecoder(value.provider_key, config, value.manifest.capability_key) for value in catalog.registrations
    )


__all__ = [
    "ARRAY_PAYLOAD_SCHEMA",
    "CONFIG_MEDIA_TYPE",
    "DEVELOPMENT_ADJUDICATOR_KEY",
    "DEVELOPMENT_GENERATOR_KEY",
    "DENOMINATOR_KEY",
    "GENERATOR_KEY",
    "HISTORY_KEY",
    'SimulatorMorphismChallengeConfigDecoder',
    'SimulatorMorphismChallengeExternalRecord',
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
    'simulator_morphism_challenges_registry',
    'simulator_morphism_challenges_phase_registry',
    'study_template',
    "protocol_template",
    "scientific_graph",
]
