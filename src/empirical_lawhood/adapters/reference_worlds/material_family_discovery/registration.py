"""Closed capability registry for the SDCB superconductor discovery child."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import (
    DiscoveryPolicyConfig,
    PolicyDecision,
    PolicyKind,
)
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .codecs import (
    MATERIAL_CORPUS_TABLE_SCHEMA,
    POLICY_HISTORY_TABLE_SCHEMA,
    WORLD_POLICY_TABLE_SCHEMA,
    WORLD_TRUTH_TABLE_SCHEMA,
)
from .contracts import MaterialFamilyConfig, MaterialSourceManifest, MaterialFamilyDiscoveryAdjudicationConfig, MaterialFamilyDiscoveryPhase, MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION, SourceQualification
from .records import PolicyHistoryPrefix, MaterialFamilyDiscoveryEvaluationFreeze, MaterialFamilyDiscoveryPhaseAdjudication, MaterialFamilyDiscoveryPhaseCloseout, MaterialFamilySourceAssessment, MaterialFamilyDiscoveryTruthControlEvaluation, MaterialFamilyDiscoveryWorldBuildConfig, MaterialFamilyDiscoveryWorldManifest, MaterialFamilyDiscoveryWorldPolicyEvaluation


CAPABILITY_VERSION = "1.0.0"
MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB = 8
SOURCE_CAPABILITY_KEY = "source.material-family-corpus"
WORLD_CAPABILITY_KEY = "reference.material-family-world"
RECEIVER_CAPABILITY_KEY = "reference.material-family-first-discovery"
FREEZE_CAPABILITY_KEY = 'transform.material-family-policy-history-freeze'
WORLD_EVALUATOR_CAPABILITY_KEY = "evaluator.material-family-first-discovery"
TRUTH_CONTROL_CAPABILITY_KEY = 'falsifier.material-family-truth-controls'
ADJUDICATOR_CAPABILITY_KEY = 'evaluator.material-family-phase-adjudication'
REPORTER_CAPABILITY_KEY = 'reporter.material-family-closeout'

_SELECTOR_KEYS = {
    PolicyKind.STRATIFIED_RANDOM: "discovery.selector.stratified-random",
    PolicyKind.MAXIMIN: "discovery.selector.maximin",
    PolicyKind.GREEDY_SCALAR_ENSEMBLE: "discovery.selector.greedy-scalar-ensemble",
    PolicyKind.BOTORCH_DISCRETE_UCB: "discovery.selector.botorch-discrete-ucb",
    PolicyKind.LOCAL_LAW_BOUNDARY: "discovery.selector.local-law-boundary",
}


def policy_capability_key(kind: PolicyKind) -> str:
    return _SELECTOR_KEYS[kind]


def capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                SOURCE_CAPABILITY_KEY,
                WORLD_CAPABILITY_KEY,
                RECEIVER_CAPABILITY_KEY,
                FREEZE_CAPABILITY_KEY,
                WORLD_EVALUATOR_CAPABILITY_KEY,
                TRUTH_CONTROL_CAPABILITY_KEY,
                ADJUDICATOR_CAPABILITY_KEY,
                REPORTER_CAPABILITY_KEY,
                *_SELECTOR_KEYS.values(),
            )
        )
    )


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode("utf-8")).hexdigest()


def _budget(
    *,
    cpu: int,
    memory_gib: int,
    wall_seconds: int,
    scan_mib: int,
    output_mib: int,
) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=cpu,
        memory_bytes=memory_gib * 1024**3,
        gpu_devices=0,
        wall_time_seconds=wall_seconds,
        source_scan_bytes=scan_mib * 1024**2,
        output_bytes=output_mib * 1024**2,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    maximum_access: OutcomeAccess,
    budget: ResourceBudget,
    checks: tuple[str, ...],
    implementation_sha256: str,
    requires_clean_commit: bool,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version=CAPABILITY_VERSION,
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=_schema_sha256(config_schema),
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=maximum_access,
        resource_ceiling=budget,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=requires_clean_commit,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def material_family_capability_registry(
    *,
    phase: MaterialFamilyDiscoveryPhase,
    implementation_sha256: str,
) -> CapabilityRegistry:
    """Build the exact phase registry without loading source or outcome bytes."""

    evaluation = phase is MaterialFamilyDiscoveryPhase.EVALUATION
    phase_access = (
        OutcomeAccess.EVALUATION_SEALED
        if evaluation
        else OutcomeAccess.DEVELOPMENT_VISIBLE
        if phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT
        else OutcomeAccess.EVALUATION_REVEALED
    )
    evaluator_access = (
        OutcomeAccess.EVALUATOR_REVEAL
        if evaluation
        else OutcomeAccess.DEVELOPMENT_VISIBLE
        if phase is MaterialFamilyDiscoveryPhase.DEVELOPMENT
        else OutcomeAccess.EVALUATION_REVEALED
    )
    read_write = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    development = (*read_write, CapabilityPermission.READ_DEVELOPMENT)
    sealed = (*read_write, CapabilityPermission.READ_SEALED_OUTCOMES)
    revealed = (*read_write, CapabilityPermission.READ_OUTCOME_VISIBLE)
    evaluator_permissions = (
        (
            *read_write,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
        if evaluation
        else development
    )
    manifests = [
        _manifest(
            key=SOURCE_CAPABILITY_KEY,
            kind=CapabilityKind.SOURCE,
            config_schema=MaterialFamilyConfig.SCHEMA,
            inputs=(
                MATERIAL_CORPUS_TABLE_SCHEMA,
                MaterialSourceManifest.SCHEMA,
                SourceQualification.SCHEMA,
            ),
            outputs=(MaterialFamilySourceAssessment.SCHEMA,),
            permissions=development,
            maximum_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            # Spawned workers import the complete pinned scientific runtime before
            # RLIMIT_AS is applied.  Its mapped address space is materially larger
            # than RSS, so every SC worker needs this common honest ceiling.
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=300,
                scan_mib=64,
                output_mib=2,
            ),
            checks=(
                'material-family-exact-source-byte-and-qualification-binding',
                'material-family-missing-operands-typed',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        _manifest(
            key=WORLD_CAPABILITY_KEY,
            kind=CapabilityKind.SIMULATOR,
            config_schema=MaterialFamilyDiscoveryWorldBuildConfig.SCHEMA,
            inputs=(MATERIAL_CORPUS_TABLE_SCHEMA, MaterialFamilySourceAssessment.SCHEMA),
            outputs=(MaterialFamilyDiscoveryWorldManifest.SCHEMA, WORLD_POLICY_TABLE_SCHEMA, WORLD_TRUTH_TABLE_SCHEMA),
            permissions=development,
            maximum_access=phase_access,
            budget=_budget(
                cpu=2,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=300,
                scan_mib=64,
                output_mib=40,
            ),
            checks=(
                'material-family-complete-family-holdout',
                'material-family-development-evaluation-material-disjoint',
                'material-family-family-and-duplicate-leakage-absent',
                'material-family-unlabelled-not-negative',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        _manifest(
            key=RECEIVER_CAPABILITY_KEY,
            kind=CapabilityKind.SIMULATOR,
            config_schema=DiscoveryPolicyConfig.SCHEMA,
            inputs=(
                PolicyDecision.SCHEMA,
                PolicyHistoryPrefix.SCHEMA,
                MaterialFamilyDiscoveryWorldManifest.SCHEMA,
                WORLD_POLICY_TABLE_SCHEMA,
                WORLD_TRUTH_TABLE_SCHEMA,
            ),
            outputs=(PolicyHistoryPrefix.SCHEMA,),
            permissions=(sealed if evaluation else development),
            maximum_access=phase_access,
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=120,
                scan_mib=64,
                output_mib=4,
            ),
            checks=(
                'material-family-batch-commitment-before-receiver-reveal',
                'material-family-policy-private-prefix-only',
                'material-family-whole-batch-cost-charged',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        *(
            (
                _manifest(
                    key=FREEZE_CAPABILITY_KEY,
                    kind=CapabilityKind.TRANSFORM,
                    config_schema=MaterialFamilyConfig.SCHEMA,
                    inputs=(PolicyHistoryPrefix.SCHEMA,),
                    outputs=(MaterialFamilyDiscoveryEvaluationFreeze.SCHEMA,),
                    permissions=(sealed if evaluation else revealed),
                    maximum_access=phase_access,
                    budget=_budget(
                        cpu=1,
                        memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                        wall_seconds=300,
                        scan_mib=512,
                        output_mib=4,
                    ),
                    checks=(
                        'material-family-complete-terminal-history-matrix',
                        'material-family-matched-policy-roster-frozen-before-reveal',
                    ),
                    implementation_sha256=implementation_sha256,
                    requires_clean_commit=evaluation,
                ),
            )
            if phase is not MaterialFamilyDiscoveryPhase.DEVELOPMENT
            else ()
        ),
        _manifest(
            key=WORLD_EVALUATOR_CAPABILITY_KEY,
            kind=(CapabilityKind.EVALUATOR if evaluation else CapabilityKind.NUMERICAL_QUALIFIER),
            config_schema=DiscoveryPolicyConfig.SCHEMA,
            inputs=(
                PolicyHistoryPrefix.SCHEMA,
                MaterialFamilyDiscoveryEvaluationFreeze.SCHEMA,
                MaterialFamilyDiscoveryWorldManifest.SCHEMA,
                WORLD_TRUTH_TABLE_SCHEMA,
            ),
            outputs=(POLICY_HISTORY_TABLE_SCHEMA, MaterialFamilyDiscoveryWorldPolicyEvaluation.SCHEMA),
            permissions=evaluator_permissions,
            maximum_access=evaluator_access,
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=120,
                scan_mib=32,
                output_mib=8,
            ),
            checks=(
                'material-family-family-unit-estimand',
                'material-family-right-censoring-retained',
                'material-family-evaluator-only-family-truth',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        _manifest(
            key=TRUTH_CONTROL_CAPABILITY_KEY,
            kind=CapabilityKind.FALSIFIER,
            config_schema=DiscoveryPolicyConfig.SCHEMA,
            inputs=(),
            outputs=(MaterialFamilyDiscoveryTruthControlEvaluation.SCHEMA,),
            permissions=read_write,
            maximum_access=OutcomeAccess.OUTCOME_BLIND,
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=120,
                scan_mib=1,
                output_mib=2,
            ),
            checks=(
                'material-family-disconnected-support-no-promotion',
                'material-family-empty-support-mandatory-hold',
                'material-family-exhausted-support-mandatory-hold',
                'material-family-zero-truth-control-false-admission',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        _manifest(
            key=ADJUDICATOR_CAPABILITY_KEY,
            kind=(
                CapabilityKind.HYPOTHESIS_ADJUDICATOR
                if evaluation
                else CapabilityKind.NUMERICAL_QUALIFIER
            ),
            config_schema=MaterialFamilyDiscoveryAdjudicationConfig.SCHEMA,
            inputs=(
                MaterialFamilyDiscoveryTruthControlEvaluation.SCHEMA,
                MaterialFamilyDiscoveryWorldPolicyEvaluation.SCHEMA,
            ),
            outputs=(MaterialFamilyDiscoveryPhaseAdjudication.SCHEMA,),
            permissions=(revealed if evaluation else development),
            maximum_access=(OutcomeAccess.EVALUATION_REVEALED if evaluation else phase_access),
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=300,
                scan_mib=64,
                output_mib=32,
            ),
            checks=(
                'material-family-paired-family-bootstrap',
                'material-family-primary-constrained-bo-gate',
                'material-family-truth-control-evidence-consumed',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
        _manifest(
            key=REPORTER_CAPABILITY_KEY,
            kind=CapabilityKind.REPORTER,
            config_schema=MaterialFamilyDiscoveryAdjudicationConfig.SCHEMA,
            inputs=(MaterialFamilyDiscoveryPhaseAdjudication.SCHEMA,),
            outputs=(MaterialFamilyDiscoveryPhaseCloseout.SCHEMA, ScientificAdjudicationRecord.SCHEMA),
            permissions=(revealed if evaluation else development),
            maximum_access=(OutcomeAccess.EVALUATION_REVEALED if evaluation else phase_access),
            budget=_budget(
                cpu=1,
                memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                wall_seconds=120,
                scan_mib=32,
                output_mib=4,
            ),
            checks=(
                'material-family-complete-world-policy-result-matrix',
                'material-family-negative-and-unevaluable-terminal-preserved',
            ),
            implementation_sha256=implementation_sha256,
            requires_clean_commit=evaluation,
        ),
    ]
    selector_permissions = sealed if evaluation else development
    for kind, key in _SELECTOR_KEYS.items():
        manifests.append(
            _manifest(
                key=key,
                kind=CapabilityKind.ANALYSIS,
                config_schema=DiscoveryPolicyConfig.SCHEMA,
                inputs=(
                    PolicyHistoryPrefix.SCHEMA,
                    MaterialFamilyDiscoveryWorldManifest.SCHEMA,
                    WORLD_POLICY_TABLE_SCHEMA,
                ),
                outputs=(PolicyDecision.SCHEMA,),
                permissions=selector_permissions,
                maximum_access=phase_access,
                budget=_budget(
                    cpu=2 if kind is PolicyKind.BOTORCH_DISCRETE_UCB else 1,
                    memory_gib=MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB,
                    wall_seconds=600 if kind is PolicyKind.BOTORCH_DISCRETE_UCB else 300,
                    scan_mib=64,
                    output_mib=2,
                ),
                checks=(
                    'material-family-deterministic-policy-replay',
                    'material-family-no-repeat-query',
                    'material-family-only-visible-prefix-consumed',
                ),
                implementation_sha256=implementation_sha256,
                requires_clean_commit=evaluation,
            )
        )
    return CapabilityRegistry(
        registry_id=f"registry.material-family-discovery-{phase.value.lower()}-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


__all__ = [
    "ADJUDICATOR_CAPABILITY_KEY",
    "CAPABILITY_VERSION",
    "FREEZE_CAPABILITY_KEY",
    "RECEIVER_CAPABILITY_KEY",
    "REPORTER_CAPABILITY_KEY",
    'MATERIAL_FAMILY_WORKER_ADDRESS_SPACE_GIB',
    "SOURCE_CAPABILITY_KEY",
    "TRUTH_CONTROL_CAPABILITY_KEY",
    "WORLD_CAPABILITY_KEY",
    "WORLD_EVALUATOR_CAPABILITY_KEY",
    "capability_keys",
    "policy_capability_key",
    'material_family_capability_registry',
]
