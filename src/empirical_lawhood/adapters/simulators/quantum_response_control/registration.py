"""Static capability registrations for the bounded quantum response control programme."""

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

from .contracts import ACQUISITION_CAPABILITY_KEY, ATLAS_CAPABILITY_KEY, CAPABILITY_VERSION, CONTROLLER_CAPABILITY_KEY, EVALUATOR_CAPABILITY_KEY, REDUCER_CAPABILITY_KEY, RESPONSE_CAPABILITY_KEY, SOURCE_CAPABILITY_KEY


CONFIG_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/quantum-trajectory-receiver-law'
CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode()).hexdigest()
PREFIX_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/prefix-checkpoint'
EVENT_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/event-table'
DRAW_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/future-draw-table'
TRANSPORT_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/transport-summary'
HANDOFF_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/development-handoff'
LAW_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/response-law'
ATLAS_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/response-atlas'
ADMISSION_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/admission'
CONTROLLER_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/controller'
RESULT_SCHEMA = 'empirical-lawhood/simulators/quantum-response-control/stage-result'


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
    wall_hours: int = 96,
    output_gib: int = 150,
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
            wall_time_seconds=wall_hours * 60 * 60,
            source_scan_bytes=2 * 1024**3,
            output_bytes=output_gib * 1024**3,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=runtime_id,
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=checks,
        implementation_sha256=implementation_sha256,
    )


def quantum_response_control_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    write = (CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,)
    read_write = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    manifests = (
        _manifest(
            key=SOURCE_CAPABILITY_KEY,
            kind=CapabilityKind.SIMULATOR,
            inputs=(),
            outputs=(PREFIX_SCHEMA, EVENT_SCHEMA, DRAW_SCHEMA, TRANSPORT_SCHEMA),
            permissions=write,
            ceiling=EvidenceCeiling.RESPONSE,
            access=OutcomeAccess.EVALUATION_SEALED,
            runtime_id="runtime.quantum-response-control.scipy-sparse-local",
            checks=(
                "quantum-response-control-action-clock-ledger",
                "quantum-response-control-checkpoint-restart",
                "quantum-response-control-fixed-hazard-source",
                "quantum-response-control-source-compatibility",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=ACQUISITION_CAPABILITY_KEY,
            kind=CapabilityKind.TRANSFORM,
            inputs=(PREFIX_SCHEMA, EVENT_SCHEMA, DRAW_SCHEMA, TRANSPORT_SCHEMA),
            outputs=(PREFIX_SCHEMA, EVENT_SCHEMA, DRAW_SCHEMA, TRANSPORT_SCHEMA),
            permissions=read_write,
            ceiling=EvidenceCeiling.RESPONSE,
            access=OutcomeAccess.EVALUATION_SEALED,
            runtime_id="runtime.quantum-response-control.sealed-acquisition",
            checks=(
                "quantum-response-control-intended-key-closure",
                "quantum-response-control-receipt-before-reveal",
                "quantum-response-control-resume-no-repeat",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=REDUCER_CAPABILITY_KEY,
            kind=CapabilityKind.OBSERVATION_OPERATOR,
            inputs=(PREFIX_SCHEMA, EVENT_SCHEMA, DRAW_SCHEMA),
            outputs=(HANDOFF_SCHEMA, TRANSPORT_SCHEMA),
            permissions=read_write,
            ceiling=EvidenceCeiling.RESPONSE,
            access=OutcomeAccess.EVALUATION_SEALED,
            runtime_id="runtime.quantum-response-control.receiver-transport",
            checks=(
                "quantum-response-control-parent-level-inference",
                "quantum-response-control-receiver-total-covariance",
                "quantum-response-control-transport-continuity",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=RESPONSE_CAPABILITY_KEY,
            kind=CapabilityKind.LAW_IDENTIFIER,
            inputs=(PREFIX_SCHEMA, DRAW_SCHEMA, TRANSPORT_SCHEMA),
            outputs=(HANDOFF_SCHEMA, LAW_SCHEMA),
            permissions=(
                CapabilityPermission.READ_DEVELOPMENT,
                *read_write,
            ),
            ceiling=EvidenceCeiling.LOCAL_LAW,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            runtime_id="runtime.quantum-response-control.record-response-law",
            checks=(
                "quantum-response-control-causal-chart",
                "quantum-response-control-fiber-pairing",
                "quantum-response-control-no-extrapolation",
                "quantum-response-control-threshold-family-types",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=ATLAS_CAPABILITY_KEY,
            kind=CapabilityKind.ATLAS_ASSEMBLER,
            inputs=(LAW_SCHEMA, HANDOFF_SCHEMA),
            outputs=(ATLAS_SCHEMA, ADMISSION_SCHEMA),
            permissions=read_write,
            ceiling=EvidenceCeiling.ADMISSION,
            access=OutcomeAccess.EVALUATION_REVEALED,
            runtime_id="runtime.quantum-response-control.atlas-admission",
            checks=(
                "quantum-response-control-atlas-gap-preservation",
                "quantum-response-control-current-path-compatibility",
                "quantum-response-control-nine-gate-intersection",
                "quantum-response-control-reachability-subset",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=CONTROLLER_CAPABILITY_KEY,
            kind=CapabilityKind.CONTROLLER_SYNTHESIZER,
            inputs=(ATLAS_SCHEMA, ADMISSION_SCHEMA),
            outputs=(CONTROLLER_SCHEMA,),
            permissions=read_write,
            ceiling=EvidenceCeiling.ADMISSION,
            access=OutcomeAccess.EVALUATION_REVEALED,
            runtime_id="runtime.quantum-response-control.action-or-hold",
            checks=(
                "quantum-response-control-controller-deadline",
                "quantum-response-control-mandatory-hold",
                "quantum-response-control-outcome-blind-commitment",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key=EVALUATOR_CAPABILITY_KEY,
            kind=CapabilityKind.EVALUATOR,
            inputs=(
                PREFIX_SCHEMA,
                EVENT_SCHEMA,
                DRAW_SCHEMA,
                HANDOFF_SCHEMA,
                CONTROLLER_SCHEMA,
            ),
            outputs=(RESULT_SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            ceiling=EvidenceCeiling.CONTROLLER_USE,
            access=OutcomeAccess.EVALUATOR_REVEAL,
            runtime_id="runtime.quantum-response-control.isolated-evaluator",
            checks=(
                "quantum-response-control-effort-matched-controls",
                "quantum-response-control-false-action-rejection",
                "quantum-response-control-separate-reveal",
                "quantum-response-control-verdict-ordering",
            ),
            implementation_sha256=implementation_sha256,
        ),
    )
    return CapabilityRegistry(
        registry_id="capabilities.quantum-response-control",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


__all__ = [
    "ADMISSION_SCHEMA",
    "ATLAS_SCHEMA",
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_SHA256",
    "CONTROLLER_SCHEMA",
    "DRAW_SCHEMA",
    "EVENT_SCHEMA",
    "HANDOFF_SCHEMA",
    "LAW_SCHEMA",
    "PREFIX_SCHEMA",
    "RESULT_SCHEMA",
    "TRANSPORT_SCHEMA",
    'quantum_response_control_registry',
]
