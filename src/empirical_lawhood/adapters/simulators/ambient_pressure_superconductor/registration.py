'Static least-privilege ambient pressure superconductor excluded solver control capability registrations.'

from __future__ import annotations

from hashlib import sha256
from typing import Any

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import ADJUDICATION_SCHEMA, ExcludedSolverControlConfig, ExcludedSolverControlSourceManifest, ExcludedSolverControlSolverSmokeResult, CAPABILITY_VERSION, CONFIG_SCHEMA, EVALUATOR_CAPABILITY_KEY, MaterialPreparationRecord, NumericalViewRecord, PREPARATION_CAPABILITY_KEY, SOLVER_CAPABILITY_KEY, SolverControlInputRecord, SolverRunObservation, decode_config_bytes
from .system import task_budget


CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode("utf-8")).hexdigest()
RUNTIME_ID = 'cpython-ambient-pressure-superconductor-excluded-solver-control-material'


class ExcludedSolverControlConfigDecoder:
    'Bind each static excluded solver control capability to one exact config byte string.'

    def __init__(self, manifest: CapabilityManifest, config: ExcludedSolverControlConfig, payload: bytes) -> None:
        if manifest.config_schema != CONFIG_SCHEMA:
            raise ValueError('excluded solver control capability uses another config schema')
        if decode_config_bytes(payload) != config:
            raise ValueError('excluded solver control registered config bytes fail strict decoding')
        self.manifest = manifest
        self.config = config
        self.payload = payload

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != CONFIG_SCHEMA or payload != self.payload:
            raise ValueError('excluded solver control config materialization differs from registration')
        if decode_config_bytes(payload) != self.config:
            raise ValueError('excluded solver control config does not satisfy its closed decoder')


def excluded_solver_control_registry(*, config: ExcludedSolverControlConfig, implementation_sha256: str) -> CapabilityRegistry:
    common: dict[str, Any] = {
        "capability_version": CAPABILITY_VERSION,
        "config_schema": CONFIG_SCHEMA,
        "config_schema_sha256": CONFIG_SCHEMA_SHA256,
        "resource_ceiling": task_budget(config),
        "deterministic": True,
        "seed_required": False,
        "language_id": "python",
        "runtime_id": RUNTIME_ID,
        "requires_clean_commit": True,
        "requires_active_mount": True,
        "requires_network": False,
        "implementation_sha256": implementation_sha256,
        "maximum_evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE,
    }
    preparation = CapabilityManifest(
        capability_key=PREPARATION_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=tuple(sorted((ExcludedSolverControlSourceManifest.SCHEMA, CONFIG_SCHEMA))),
        output_schema_ids=tuple(
            sorted(
                (
                    MaterialPreparationRecord.SCHEMA,
                    NumericalViewRecord.SCHEMA,
                    SolverControlInputRecord.SCHEMA,
                )
            )
        ),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            "exact-held-pb-preparation",
            "hold-only-action-history",
            "target-contact-count-zero",
        ),
        **common,
    )
    solver = CapabilityManifest(
        capability_key=SOLVER_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    MaterialPreparationRecord.SCHEMA,
                    NumericalViewRecord.SCHEMA,
                    SolverControlInputRecord.SCHEMA,
                )
            )
        ),
        output_schema_ids=(SolverRunObservation.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        conformance_check_ids=(
            "bounded-qe-output-extraction",
            "fixed-executor-profile",
            "network-disabled",
            "source-input-pseudo-binary-digest-binding",
        ),
        **common,
    )
    evaluator = CapabilityManifest(
        capability_key=EVALUATOR_CAPABILITY_KEY,
        kind=CapabilityKind.EVALUATOR,
        input_schema_ids=tuple(
            sorted((SolverControlInputRecord.SCHEMA, SolverRunObservation.SCHEMA))
        ),
        output_schema_ids=tuple(sorted((ADJUDICATION_SCHEMA, ExcludedSolverControlSolverSmokeResult.SCHEMA))),
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
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        conformance_check_ids=(
            "excluded-control-nonpromotable",
            "required-operand-intersection",
            "target-contact-count-zero",
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id='capabilities.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke',
        capabilities=tuple(
            sorted((preparation, solver, evaluator), key=lambda value: value.registry_id)
        ),
    )


def config_decoders(
    *, registry: CapabilityRegistry, config: ExcludedSolverControlConfig, payload: bytes
) -> tuple[ExcludedSolverControlConfigDecoder, ...]:
    return tuple(ExcludedSolverControlConfigDecoder(manifest, config, payload) for manifest in registry.capabilities)


__all__ = [
    'ExcludedSolverControlConfigDecoder',
    "CONFIG_SCHEMA_SHA256",
    "RUNTIME_ID",
    'excluded_solver_control_registry',
    "config_decoders",
]
