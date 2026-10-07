"""Static direct-TORAX flagship capability catalog."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.adapters.physical.mast_archive.contracts import MastCausalState

from .contracts import (
    ToraxMappingResult,
    ToraxMatchedPanel,
    ToraxPreparation,
)
from .mapping import ToraxMappingConfig
from .evaluation import ToraxAssumptionEnsembleEvaluation, ToraxEvaluationConfig


def build_torax_control_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    mapping = CapabilityManifest(
        capability_key="torax-flagship.mast-preparation-mapper",
        capability_version="1.0.0",
        kind=CapabilityKind.TRANSFORM,
        config_schema=ToraxMappingConfig.SCHEMA,
        config_schema_sha256=sha256(ToraxMappingConfig.SCHEMA.encode()).hexdigest(),
        input_schema_ids=(MastCausalState.SCHEMA,),
        output_schema_ids=(ToraxMappingResult.SCHEMA,),
        permissions=(CapabilityPermission.READ_DEVELOPMENT,),
        maximum_evidence_ceiling=EvidenceCeiling.RESPONSE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=ResourceBudget(
            cpu_cores=2,
            memory_bytes=4 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=600,
            source_scan_bytes=0,
            output_bytes=64 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-torax-mapping",
        requires_clean_commit=False,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=(
            "field-provenance-classified",
            "stock-iter-hybrid-forbidden",
            "typed-mapping-rejection",
        ),
        implementation_sha256=implementation_sha256,
    )
    runner = CapabilityManifest(
        capability_key="torax-flagship.matched-panel-runner",
        capability_version="1.0.0",
        kind=CapabilityKind.SIMULATOR,
        config_schema=ToraxPreparation.SCHEMA,
        config_schema_sha256=sha256(ToraxPreparation.SCHEMA.encode()).hexdigest(),
        input_schema_ids=(ToraxPreparation.SCHEMA,),
        output_schema_ids=(ToraxMatchedPanel.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=ResourceBudget(
            cpu_cores=16,
            memory_bytes=64 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=8 * 3600,
            source_scan_bytes=4 * 1024**3,
            output_bytes=64 * 1024**3,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="torax-1.4.2-cpu-x64-flagship",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "direct-torax-native-runtime",
            "down-hold-up-complete-panel",
            "invalid-rollout-not-scoreable",
            "runtime-artifacts-external",
        ),
        implementation_sha256=implementation_sha256,
    )
    evaluator = CapabilityManifest(
        capability_key="torax-flagship.panel-evaluator",
        capability_version="1.0.0",
        kind=CapabilityKind.EVALUATOR,
        config_schema=ToraxEvaluationConfig.SCHEMA,
        config_schema_sha256=sha256(ToraxEvaluationConfig.SCHEMA.encode()).hexdigest(),
        input_schema_ids=(ToraxMatchedPanel.SCHEMA,),
        output_schema_ids=(ToraxAssumptionEnsembleEvaluation.SCHEMA,),
        permissions=(CapabilityPermission.READ_DEVELOPMENT,),
        maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=4 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=1800,
            source_scan_bytes=0,
            output_bytes=128 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-torax-panel-evaluator",
        requires_clean_commit=False,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=(
            "incomplete-panels-unevaluable",
            "preparation-first-aggregation",
            "theta-and-view-nesting-retained",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="torax-flagship-registry",
        capabilities=(mapping, runner, evaluator),
    )


__all__ = ['build_torax_control_registry']
