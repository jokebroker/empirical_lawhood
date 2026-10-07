"""Static capability registrations for the bounded quantum receiver response programme."""

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

from .contracts import CAPABILITY_VERSION, EVALUATOR_CAPABILITY_KEY, METHOD_CAPABILITY_KEY, SOURCE_CAPABILITY_KEY


CONFIG_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/quantum-trajectory-receiver-law'
CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode()).hexdigest()

PREFIX_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/prefix-checkpoint'
EVENT_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/event-table'
DRAW_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/future-draw-table'
HANDOFF_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/development-handoff'
RESULT_SCHEMA = 'empirical-lawhood/simulators/quantum-receiver-response/stage-result'


def quantum_receiver_response_registry(*, implementation_sha256: str) -> CapabilityRegistry:
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
        output_schema_ids=tuple(sorted((DRAW_SCHEMA, EVENT_SCHEMA, PREFIX_SCHEMA))),
        permissions=tuple(sorted((CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,))),
        maximum_evidence_ceiling=EvidenceCeiling.RESPONSE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-receiver-response.scipy-sparse-local",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "quantum-receiver-response-basis-operator-conformance",
            "quantum-receiver-response-checkpoint-restart",
            "quantum-receiver-response-dense-sparse-pathwise",
            "quantum-receiver-response-fixed-hazard-unravelling",
        ),
        implementation_sha256=implementation_sha256,
    )
    method = CapabilityManifest(
        capability_key=METHOD_CAPABILITY_KEY,
        capability_version=CAPABILITY_VERSION,
        kind=CapabilityKind.LAW_IDENTIFIER,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=tuple(sorted((DRAW_SCHEMA, EVENT_SCHEMA, PREFIX_SCHEMA))),
        output_schema_ids=(HANDOFF_SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-receiver-response.record-law-local",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "quantum-receiver-response-causal-cutoff",
            "quantum-receiver-response-chart-selection",
            "quantum-receiver-response-support-no-extrapolation",
            "quantum-receiver-response-threshold-family-separation",
        ),
        implementation_sha256=implementation_sha256,
    )
    evaluator = CapabilityManifest(
        capability_key=EVALUATOR_CAPABILITY_KEY,
        capability_version=CAPABILITY_VERSION,
        kind=CapabilityKind.EVALUATOR,
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=CONFIG_SCHEMA_SHA256,
        input_schema_ids=tuple(sorted((DRAW_SCHEMA, EVENT_SCHEMA, HANDOFF_SCHEMA, PREFIX_SCHEMA))),
        output_schema_ids=(RESULT_SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="runtime.quantum-receiver-response.isolated-evaluator",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "quantum-receiver-response-evaluation-sealed",
            "quantum-receiver-response-preparation-bootstrap",
            "quantum-receiver-response-receiver-total-covariance",
            "quantum-receiver-response-verdict-ordering",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="capabilities.quantum-receiver-response",
        capabilities=tuple(
            sorted(
                (source, method, evaluator),
                key=lambda value: value.registry_id,
            )
        ),
    )


__all__ = [
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_SHA256",
    "DRAW_SCHEMA",
    "EVENT_SCHEMA",
    "HANDOFF_SCHEMA",
    "PREFIX_SCHEMA",
    "RESULT_SCHEMA",
    'quantum_receiver_response_registry',
]
