"""Static simulator discovery; no native imports, source reads or probes."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .panel import ReactorPrefixConfig, ReactorPrefixPanel, ReactorSourceBundle


CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science.reactor-prefix",
    kind=CapabilityKind.SIMULATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    config_schema=ReactorPrefixConfig.SCHEMA,
    config_schema_sha256=sha256(ReactorPrefixConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted((ReactorPrefixConfig.SCHEMA, ReactorSourceBundle.SCHEMA))),
    output_schema_ids=(ReactorPrefixPanel.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 300, 1024**2, 1024**2),
    implementation_sha256=sha256(
        b"reactor-prefix-response:native-trace-two-callback-delivery"
    ).hexdigest(),
    runtime_id="cpython-3.11.14-numpy-2.4.6-tbs-reactor-prefix",
    conformance_check_ids=(
        "exact-source-pin",
        "no-inferred-delivery",
        "two-decision-causal-barrier",
        "whole-unit-view-roster",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "reactor-prefix",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ReactorPrefixConfig,),
    None,
    namespace="reactor-prefix-response",
)
