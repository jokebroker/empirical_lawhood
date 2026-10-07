"""Import-safe discovery for one model and two nested native RC views."""

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

from .contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderStudyConfig

CAPABILITY = replace(
    FINITE_ACTION_MANIFEST,
    capability_key="rc-ladder-response.native-view",
    kind=CapabilityKind.SIMULATOR,
    config_schema=ResistorCapacitorLadderStudyConfig.SCHEMA,
    config_schema_sha256=sha256(ResistorCapacitorLadderStudyConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(ResistorCapacitorLadderStudyConfig.SCHEMA,),
    output_schema_ids=(ResistorCapacitorLadderNativePanel.SCHEMA,),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(1, 1024**3, 0, 120, 128 * 1024, 1024**2),
    implementation_sha256=sha256(
        b"rc-ladder-response:source-impedance-piecewise-matrix-and-backward-euler"
    ).hexdigest(),
    runtime_id="cpython-3.11-numpy-2.4.6-scipy-1.17.1-rc-ladder-response",
    conformance_check_ids=(
        "frozen-native-model-and-action",
        "one-model-two-nested-views",
        "output-grid-charge-balance",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, COMPONENTS = bundle(
    "native-view",
    ExtensionContributionKind.SOURCE,
    CAPABILITY,
    (ResistorCapacitorLadderStudyConfig,),
    None,
    namespace="rc-ladder-response",
)
