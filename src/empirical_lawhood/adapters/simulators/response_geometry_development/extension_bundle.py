"""Interactive native D source; this never advertises a file dataset."""

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.runtime.extension_bundles import ExtensionContributionKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.simulators.response_geometry_prospective.discovery import bundle, manifest
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID, DEVELOPMENT_HDF5_SCHEMA, ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeSegmentResult

DEVELOPMENT_SOURCE_CAPABILITY = manifest(
    "source",
    ResponseGeometryDevelopmentNativeConfig,
    CapabilityKind.SIMULATOR,
    (
        ResponseGeometryDevelopmentNativeConfig.SCHEMA,
        ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA,
        DEVELOPMENT_HDF5_SCHEMA,
        LinkedCampaignStageEnvelope.SCHEMA,
    ),
    (ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA, DEVELOPMENT_HDF5_SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
    EvidenceCeiling.MEASUREMENT,
    OutcomeAccess.DEVELOPMENT_VISIBLE,
    1800,
    namespace=DEVELOPMENT_PANEL_ID,
    resources=ResourceBudget(1, 3 * 1024**3, 0, 1800, 32 * 1024**2, 17 * 1024**2),
    read_permission=CapabilityPermission.READ_DEVELOPMENT,
    implementation_id="response-geometry-development-short-pulse-native-panel",
    conformance_ids=(
        "exact-d-native-roster",
        "no-scientific-retry",
        "separate-root-view-action-clocks",
    ),
)
EXTENSION_BUNDLE_CONTRIBUTION, DEVELOPMENT_SOURCE_COMPONENTS = bundle(
    "source",
    ExtensionContributionKind.SOURCE,
    DEVELOPMENT_SOURCE_CAPABILITY,
    (ResponseGeometryDevelopmentNativeConfig,),
    DEVELOPMENT_HDF5_SCHEMA,
    namespace=DEVELOPMENT_PANEL_ID,
)
