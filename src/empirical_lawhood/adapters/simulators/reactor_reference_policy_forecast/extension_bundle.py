"""Static simulator discovery; no native imports, source reads or probes."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchConfig, ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import ReactorBatchTrace, MAXIMUM_BATCH_BYTES


CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science-batch.reactor-batch",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    config_schema=ReactorBatchConfig.SCHEMA,
    config_schema_sha256=sha256(ReactorBatchConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted((ReactorBatchConfig.SCHEMA, ReactorBatchSource.SCHEMA))),
    output_schema_ids=(ReactorBatchTrace.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    resource_ceiling=ResourceBudget(1, 4 * 1024**3, 0, 300, 1024**2, MAXIMUM_BATCH_BYTES),
    implementation_sha256=sha256(
        b"reactor-batch-native:causal-observer-forecast-and-native-full-batch"
    ).hexdigest(),
    runtime_id="cpython-3.11.14-numpy-2.4.6-reactor-batch-native",
    conformance_check_ids=(
        "2880-decision-causal-barriers",
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
    namespace="reactor-reference-policy-forecast",
)
