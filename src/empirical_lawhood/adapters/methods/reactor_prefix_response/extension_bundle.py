"""Noncontact discovery of the task's finite-action campaign binding."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import FINITE_ACTION_MANIFEST
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import ReactorPrefixPanel
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from .design import ReactorScienceDesign
from .result import ReactorScienceResult


INPUT_TYPES = (
    ReactorScienceDesign,
    ReactorPrefixPanel,
)
OUTPUT_TYPES = (
    ReactorScienceResult,
    ScientificAdjudicationRecord,
)
CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="terminal-bench-science.finite-chain",
    kind=CapabilityKind.EVALUATOR,
    maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    permissions=tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            ),
            key=lambda p: p.value,
        )
    ),
    config_schema=ReactorScienceDesign.SCHEMA,
    config_schema_sha256=sha256(ReactorScienceDesign.SCHEMA.encode()).hexdigest(),
    input_schema_ids=tuple(sorted(t.SCHEMA for t in INPUT_TYPES)),
    output_schema_ids=tuple(sorted(t.SCHEMA for t in OUTPUT_TYPES)),
    implementation_sha256=sha256(
        b"reactor-finite-chain:existing-producer-profile-sole-finalizer"
    ).hexdigest(),
    runtime_id="cpython-3.11-finite-reactor-chain",
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 300, 8 * 1024**2, 24 * 1024**2),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "finite-chain",
    ExtensionContributionKind.METHOD,
    CAPABILITY,
    (ReactorScienceDesign,),
    None,
    namespace="reactor-prefix-response",
)
