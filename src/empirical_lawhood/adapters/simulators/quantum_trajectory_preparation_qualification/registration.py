"""Static bounded capability registry for quantum trajectory preparation qualification."""

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

from .types import CAPABILITY_VERSION, EVALUATOR_CAPABILITY_KEY, METHOD_CAPABILITY_KEY, SOURCE_CAPABILITY_KEY


CONFIG_SCHEMA = 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/quantum-trajectory-source-qualification'
CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode()).hexdigest()
TRAJECTORY_SCHEMA = 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/trajectory'
OBSERVATION_SCHEMA = 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/observation'
HANDOFF_SCHEMA = 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/development-handoff'
RESULT_SCHEMA = 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/stage-result'


def quantum_trajectory_preparation_qualification_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    resources = ResourceBudget(
        cpu_cores=8,
        memory_bytes=16 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=24 * 60 * 60,
        source_scan_bytes=0,
        output_bytes=20 * 1024**3,
    )
    source = CapabilityManifest(
        capability_key=SOURCE_CAPABILITY_KEY,
        capability_version=CAPABILITY_VERSION,
        kind=CapabilityKind.SIMULATOR,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=(),
        output_schema_ids=(TRAJECTORY_SCHEMA,),
        permissions=(CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-trajectory-preparation-qualification.scipy-local",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "source-conformance-typed-jump-contract",
            "quantum-trajectory-preparation-qualification-ensemble_conformance-empirical-bernstein",
            "quantum-trajectory-preparation-qualification-ensemble_conformance-lindblad-unravelling",
            "path-conformance-dense-sparse-path",
            "path-conformance-exact-restart",
            "deterministic-reference-direct-dop853-reference",
            "deterministic-reference-sparse-kron-reference",
        ),
        implementation_sha256=implementation_sha256,
    )
    method = CapabilityManifest(
        capability_key=METHOD_CAPABILITY_KEY,
        capability_version=CAPABILITY_VERSION,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=(OBSERVATION_SCHEMA, TRAJECTORY_SCHEMA),
        output_schema_ids=(HANDOFF_SCHEMA,),
        permissions=(
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-trajectory-preparation-qualification.stationarity-selection",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "quantum-trajectory-preparation-qualification-joint-12-look-parent-bootstrap",
            "quantum-trajectory-preparation-qualification-preparation-memory",
            "statistical-conformance-joint-statistical-instrument",
            "quantum-trajectory-preparation-qualification-stable-replication-suffix",
            "quantum-trajectory-preparation-qualification-threshold-nonretuning",
        ),
        implementation_sha256=implementation_sha256,
    )
    evaluator = CapabilityManifest(
        capability_key=EVALUATOR_CAPABILITY_KEY,
        capability_version=CAPABILITY_VERSION,
        kind=CapabilityKind.EVALUATOR,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=(HANDOFF_SCHEMA, OBSERVATION_SCHEMA, TRAJECTORY_SCHEMA),
        output_schema_ids=(RESULT_SCHEMA,),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-trajectory-preparation-qualification.sealed-evaluator",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "quantum-trajectory-preparation-qualification-evaluation-roster-sealed",
            "quantum-trajectory-preparation-qualification-evaluation-template-slots",
            "quantum-trajectory-preparation-qualification-joint-grid-simultaneous-intervals",
            "quantum-trajectory-preparation-qualification-noncompensating-verdict",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="capabilities.quantum-trajectory-preparation-qualification",
        capabilities=tuple(
            sorted((source, method, evaluator), key=lambda value: value.registry_id)
        ),
    )


__all__ = [
    "CONFIG_SCHEMA",
    "HANDOFF_SCHEMA",
    "OBSERVATION_SCHEMA",
    "RESULT_SCHEMA",
    "TRAJECTORY_SCHEMA",
    'quantum_trajectory_preparation_qualification_registry',
]
