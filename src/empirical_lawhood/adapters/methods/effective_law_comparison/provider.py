"""Static capability registration for the effective-law post-hoc tranche."""

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
from empirical_lawhood.runtime.posthoc import (
    ANALYSIS_RESULT_SCHEMA,
    INTEGRATED_SKEPTIC_SCHEMA,
    METATHEORY_SYNTHESIS_SCHEMA,
    POSTHOC_CAPABILITY_KEY,
    POSTHOC_CAPABILITY_VERSION,
    POSTHOC_CONFIG_SCHEMA,
    RESULT_INDEX_SCHEMA,
    SKEPTIC_RESULT_SCHEMA,
)

SUPPORTED_INPUT_SCHEMAS = tuple(
    sorted(
        {
            'empirical-lawhood/adapters/simulators/denominator-response-qualification-stop-handoff',
            'empirical-lawhood/adapters/simulators/staged-hybrid-response-lane-formal-panel',
            'empirical-lawhood/adapters/simulators/battery-electrothermal-adjudication',
            'empirical-lawhood/adapters/simulators/battery-exact-restart-adjudication',
            'empirical-lawhood/simulators/battery-reduced-observation-prediction/reduced-coordinate-adjudication',
            'empirical-lawhood/simulators/prepared-base-qualification-receipt',
            'empirical-lawhood/adapters/simulators/numerical-denominator-evaluation-adjudication',
            'empirical-lawhood/methods/target-construct-validation/target-construct-validation-categorical-forecast-panel',
            'empirical-lawhood/methods/target-construct-validation/target-construct-validation-cross-target-interpretation-correction',
            'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-evaluation-bundle',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-comparator-encoding',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-selection',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-forecast-challenge-qualification',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-forecast-separation-report',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-power-design-qualification',
            'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel',
            'empirical-lawhood/independent-substrate-grounding/independent-substrate-axis-adjudication',
            'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-terminal-handoff',
            "empirical-lawhood/methods/predictive-recurrence-adjudication",
            'empirical-lawhood/methods/action-fiber-structural-recurrence-adjudication',
            'empirical-lawhood/methods/margin-structural-recurrence-forecast-adjudication',
            'empirical-lawhood/runtime/scientific-adjudication-record',
            ANALYSIS_RESULT_SCHEMA,
            INTEGRATED_SKEPTIC_SCHEMA,
            METATHEORY_SYNTHESIS_SCHEMA,
            SKEPTIC_RESULT_SCHEMA,
        }
    )
)


def posthoc_capability_manifest(implementation_sha256: str) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=POSTHOC_CAPABILITY_KEY,
        capability_version=POSTHOC_CAPABILITY_VERSION,
        kind=CapabilityKind.ANALYSIS,
        config_schema=POSTHOC_CONFIG_SCHEMA,
        config_schema_sha256=sha256(POSTHOC_CONFIG_SCHEMA.encode("utf-8")).hexdigest(),
        input_schema_ids=SUPPORTED_INPUT_SCHEMAS,
        output_schema_ids=tuple(
            sorted(
                (
                    ANALYSIS_RESULT_SCHEMA,
                    INTEGRATED_SKEPTIC_SCHEMA,
                    METATHEORY_SYNTHESIS_SCHEMA,
                    RESULT_INDEX_SCHEMA,
                    SKEPTIC_RESULT_SCHEMA,
                )
            )
        ),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_OUTCOME_VISIBLE,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=ResourceBudget(
            cpu_cores=8,
            memory_bytes=32 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=6 * 60 * 60,
            source_scan_bytes=4 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-effective-law-comparison",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "complete-unit-independence",
            "counterexample-precedence",
            "no-cross-partition-pooling",
            "outcome-visible-nonpromotion",
            "typed-negative-and-unevaluable",
        ),
        implementation_sha256=implementation_sha256,
    )


def posthoc_capability_registry(implementation_sha256: str) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="effective-law-comparison",
        capabilities=(posthoc_capability_manifest(implementation_sha256),),
    )


__all__ = [
    "SUPPORTED_INPUT_SCHEMAS",
    "posthoc_capability_manifest",
    "posthoc_capability_registry",
]
