"""Discoverable synthetic uniform electron gas analytic acquisition, separate from truth-known conformance."""

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.composition.discovery import bundle
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind

from .analytic_contracts import UniformElectronGasAnalyticPanel
from .native_quickstart import UniformElectronGasAnalyticReferenceConfig

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="uniform-electron-gas-analytic-reference.native-panel",
    kind=CapabilityKind.SIMULATOR,
    config_schema=UniformElectronGasAnalyticReferenceConfig.SCHEMA,
    config_schema_sha256=sha256(UniformElectronGasAnalyticReferenceConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(UniformElectronGasAnalyticReferenceConfig.SCHEMA,),
    output_schema_ids=(UniformElectronGasAnalyticPanel.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 120, 128 * 1024, 1024**2),
    implementation_sha256=sha256(
        b"uniform-electron-gas-analytic-reference:one-synthetic-si-q-action-slab-panel"
    ).hexdigest(),
    runtime_id="cpython-3.11-scipy-1.17.1-uniform-electron-gas-analytic-reference",
    conformance_check_ids=(
        "one-synthetic-acquisition-twenty-nested-conditions",
        "requested-accepted-applied-realized-si-action",
        "static-transverse-current-and-slab-profile",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "analytic-panel",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (UniformElectronGasAnalyticReferenceConfig,),
    None,
    namespace="uniform-electron-gas-analytic-reference",
)
