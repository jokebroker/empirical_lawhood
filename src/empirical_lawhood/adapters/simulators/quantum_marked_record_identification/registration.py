"""Closed static capability registry for the bounded quantum marked record identification tranche."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import CAPABILITY_VERSION, DEVELOPMENT_CAPABILITY_KEY, EVALUATOR_CAPABILITY_KEY, NOMINATION_CAPABILITY_KEY, SOURCE_CAPABILITY_KEY


CONFIG_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/quantum-trajectory-receiver-law'
CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode()).hexdigest()
PREFIX_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/prefix-checkpoint'
EVENT_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/event-table'
FUTURE_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/passive-future'
NOMINATION_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/nomination'
COMPILER_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/score-compiler'
RESULT_SCHEMA = 'empirical-lawhood/simulators/quantum-marked-record-identification/result'


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    ceiling: EvidenceCeiling,
    access: OutcomeAccess,
    runtime_id: str,
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version=CAPABILITY_VERSION,
        kind=kind,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=ceiling,
        maximum_outcome_access=access,
        resource_ceiling=ResourceBudget(
            cpu_cores=8,
            memory_bytes=32 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=72 * 60 * 60,
            source_scan_bytes=8 * 1024**3,
            output_bytes=100 * 1024**3,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=runtime_id,
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def quantum_marked_record_identification_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    read_write = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    manifests = (
        _manifest(
            key=SOURCE_CAPABILITY_KEY,
            kind=CapabilityKind.SIMULATOR,
            inputs=(),
            outputs=(EVENT_SCHEMA, FUTURE_SCHEMA, PREFIX_SCHEMA),
            permissions=(CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,),
            ceiling=EvidenceCeiling.ORDER_RELATION,
            access=OutcomeAccess.EVALUATION_SEALED,
            runtime_id="runtime.quantum-marked-record-identification.passive-source",
            checks=(
                "quantum-marked-record-identification-cutoff-ordering",
                "quantum-marked-record-identification-prefix-before-future",
                "quantum-marked-record-identification-source-compatibility",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=NOMINATION_CAPABILITY_KEY,
            kind=CapabilityKind.PROSPECTIVE_NOMINATOR,
            inputs=(EVENT_SCHEMA,),
            outputs=(NOMINATION_SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            ceiling=EvidenceCeiling.NON_PROMOTABLE,
            access=OutcomeAccess.EVALUATION_REVEALED,
            runtime_id="runtime.quantum-marked-record-identification.outcome-visible-nomination",
            checks=("quantum-marked-record-identification-finite-grammar", "quantum-marked-record-identification-nomination-nonpromotion"),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=DEVELOPMENT_CAPABILITY_KEY,
            kind=CapabilityKind.ANALYSIS,
            inputs=(EVENT_SCHEMA, FUTURE_SCHEMA, NOMINATION_SCHEMA),
            outputs=(COMPILER_SCHEMA,),
            permissions=(
                CapabilityPermission.READ_DEVELOPMENT,
                *read_write,
            ),
            ceiling=EvidenceCeiling.NON_PROMOTABLE,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            runtime_id="runtime.quantum-marked-record-identification.development-compiler",
            checks=(
                "quantum-marked-record-identification-fit-selection-separation",
                "quantum-marked-record-identification-transparent-ridge",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=EVALUATOR_CAPABILITY_KEY,
            kind=CapabilityKind.EVALUATOR,
            inputs=(COMPILER_SCHEMA, EVENT_SCHEMA, FUTURE_SCHEMA),
            outputs=(RESULT_SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            ceiling=EvidenceCeiling.ORDER_RELATION,
            access=OutcomeAccess.EVALUATOR_REVEAL,
            runtime_id="runtime.quantum-marked-record-identification.isolated-evaluator",
            checks=(
                "quantum-marked-record-identification-parent-bootstrap",
                "quantum-marked-record-identification-separate-reveal",
                "quantum-marked-record-identification-verdict-precedence",
            ),
            implementation_sha256=implementation_sha256,
        ),
    )
    return CapabilityRegistry(
        registry_id="capabilities.quantum-marked-record-identification",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


__all__ = [
    "COMPILER_SCHEMA",
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_SHA256",
    "EVENT_SCHEMA",
    "FUTURE_SCHEMA",
    "NOMINATION_SCHEMA",
    "PREFIX_SCHEMA",
    "RESULT_SCHEMA",
    'quantum_marked_record_identification_registry',
]
