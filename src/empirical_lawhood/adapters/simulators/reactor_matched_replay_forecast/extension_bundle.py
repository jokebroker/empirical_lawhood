"""Static simulator discovery; no native imports, source reads or probes."""

from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.transport import ReactorTraceEnvelope

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.design import ReactorBatchConfig, ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import MAXIMUM_BATCH_BYTES


CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="reactor-matched-replay-forecast.reactor-batch",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    config_schema=ReactorBatchConfig.SCHEMA,
    config_schema_sha256=sha256(ReactorBatchConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted((ReactorBatchConfig.SCHEMA, ReactorBatchSource.SCHEMA))),
    output_schema_ids=(ReactorTraceEnvelope.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    resource_ceiling=ResourceBudget(1, 4 * 1024**3, 0, 1200, 1024**2, MAXIMUM_BATCH_BYTES),
    implementation_sha256=sha256(
        b"reactor-matched-replay-forecast-programme:matched-input-policy-robustness-and-prescribed-recovery"
    ).hexdigest(),
    runtime_id="cpython-3.11.14-numpy-2.4.6-reactor-matched-replay-forecast-programme",
    conformance_check_ids=(
        "14400-decisions-five-nested-episodes",
        "exact-source-pin",
        "no-inferred-delivery",
        "whole-unit-view-roster",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "reactor-batch",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ReactorBatchConfig,),
    None,
    namespace="reactor-matched-replay-forecast",
)
